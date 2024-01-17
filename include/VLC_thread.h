#ifndef _VLC_H_
#define _VLC_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <thread>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <seccomp.h>
#include <sched.h>

#ifdef NDEBUG
#define DEBUG(x) 
#else
#define DEBUG(x) do { std::cerr << "[Debug] " << x << std::endl; } while (0)
#endif

#define STACK_SIZE (1024 * 1024 * 8)

namespace VLC {

pid_t MONITOR_PID;  // a global variable that application thread could read

class Runtime {
public:
    Runtime() = default;

    ~Runtime() = default;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    /**
     * intialization VLC runtime.
     * 
     * A fork will happen, where parent process becomes VLC manager,
     * application continues on child process.
     * 
     * Manager process will not exit until application process is finished.
    */
    void initialize(int (*fn)(void *), void *arg) {
        MONITOR_PID = getpid();

        // create a new thread, which will become the application main thread
        // allocate stack for new thread
        stack = mmap(NULL, STACK_SIZE, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
        if (stack == MAP_FAILED) {
            std::cerr << "VLC: unable to malloc application stack, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        launch_application_arg application = {fn, arg};

        // only works for x86-64
        // new thread will have different pid but shared VM, FS, FD, IO
        if (clone(launch_application, (char *) stack + STACK_SIZE, CLONE_VM | CLONE_IO | CLONE_FS | CLONE_FILES | CLONE_PARENT_SETTID,
            (void *) &application, &application_pid, NULL, NULL) == -1) {
            std::cerr << "VLC: unable to create new thread for application, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // this is manager process (parent)
        std::cout << "VLC: manager start." << std::endl;
        std::cout << application_pid << std::endl;
        determine_resouces();

        // let application terminate along with monitor process
        ptrace(PTRACE_SETOPTIONS, application_pid, 0, PTRACE_O_EXITKILL);

        // wait on application seccomp bpf configuration
        int status;
        if (wait(&status) == -1) {
            std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // trace on seccomp signal
        ptrace(PTRACE_SETOPTIONS, application_pid, 0, PTRACE_O_TRACESECCOMP);
        ptrace(PTRACE_CONT, application_pid, 0, 0);

        // start monitoring the application
        intercept_syscall();

        // exit when the application finished
        std::exit(EXIT_SUCCESS);  
        
    }

private:
    cpu_set_t system_cpu_set;
    cpu_set_t virtual_cpu_set;
    bool virtual_cpu_set_is_set = false;
    pid_t application_pid = 0;
    void *stack;

    struct launch_application_arg {
        int (*fn)(void *);
        void *arg;
    };

    static int launch_application(void *arg) {
        // this is application thread (child)
        // set a BPF filter on specific syscall for ptrace
        configure_seccomp_bpf();
        raise(SIGSTOP); // stop and let monitor process know

        // return to the application's main()
        std::cout << "VLC: application start." << std::endl;
        launch_application_arg *application = (launch_application_arg *) arg;
        return application->fn(application->arg);
    }

    /**
     * Configure a BPF filter for ptrace + seccomp.
     * 
     * Will mark the caller process as tracee.
     * This allows the tracer only stop on specific syscall
     * that related to resouces query rather than all of them.
     * So the overhead of interception could be minimized.
    */
   static void configure_seccomp_bpf() {
        // BFP program body
        struct sock_filter filter[] = {
            // load syscall number
            BPF_STMT(BPF_LD + BPF_W + BPF_ABS, offsetof(struct seccomp_data, nr)),
            // if it is sched_getaffinity(), return TRACE
            BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, SYS_sched_getaffinity, 0, 1),
            BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_TRACE),
            // else, continue the syscall without tracing
            BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_ALLOW),
        };

        struct sock_fprog prog = {
            .len = (unsigned short) (sizeof(filter)/sizeof(filter[0])),
            .filter = filter
        };

        ptrace(PTRACE_TRACEME, 0, 0, 0); // mark this process as tracee

        // promise not to grant any new privileges
        // so no need to run this program with root
        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == -1) {
            std::cerr << "VLC: unable to promise no_new_privs" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // set seccomp with the above BFP program
        if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) == -1) {
            std::cerr << "VLC: untable to set seccomp filter, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
   }

    /**
     * Query this system's resouce limits,
     * including cpu set
    */
    void determine_resouces() {
        // cpu set
        if (sched_getaffinity(0, sizeof(cpu_set_t), &system_cpu_set) == -1) {
            std::cerr << "VLC: unable to determine cpu set, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    /**
     * A ptrace loop that wait application to stop on syscall
     * Will modify the result to achieve resouce virtualization
    */
    void intercept_syscall() {
        int status;
        while (true) {
            // TODO: waitpid(-1, &status, __WALL) for multithread
            if (wait(&status) == -1) {
                std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            
            DEBUG("VLC: stop application.");

            // stopped on seccomp signal
            if (status >> 8 == (SIGTRAP | (PTRACE_EVENT_SECCOMP << 8))) {
                // we alreay at the syscall entry point,
                // so continue until syscall exit
                if (ptrace(PTRACE_SYSCALL, application_pid, 0, 0) == -1)  {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (wait(&status) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                
                // retrive syscall arguments
                user_regs_struct regs;
                if (ptrace(PTRACE_GETREGS, application_pid, 0, &regs) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                long syscall = regs.orig_rax;

                if (syscall == SYS_sched_getaffinity) {  // capture sys_sched_getaffinity
                    forge_sched_getaffinity(regs.rdx);
                }
            }

            if (WIFEXITED(status) || WIFSIGNALED(status)) {
                // child has exited or terminated
                std::cout << "VLC: manager exit." << std::endl;
                break;
            }

            // continue until next seccomp signal
            if (ptrace(PTRACE_CONT, application_pid, 0, 0) == -1) {
                std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * modify the result of sys_sched_getaffinity
     * will replace original content in `user_mask_ptr`
     * with a virtualized cpu affinity.
     * 
     * @param user_mask_ptr a remote pointer to application's cpu set buffer
     * Its content will be modified.
    */
    void forge_sched_getaffinity(unsigned long user_mask_ptr) {
        // TODO: add policy here
        // make a virtual cpu affinity
        if (!virtual_cpu_set_is_set) {
            virtual_cpu_set = system_cpu_set;
            
            for (int i = 0; i < 24; i++) {
                CPU_CLR(i, &virtual_cpu_set);
            }
            virtual_cpu_set_is_set = true;
        }
        
        *((cpu_set_t *) user_mask_ptr) = virtual_cpu_set;

        DEBUG("VLC: sched_getaffinity() is modifed.");
    }
};
    
// void enable() {
//     std::thread::id this_id = std::this_thread::get_id();
// }

// void register_thread(pthread_t thread, int id);

}
#endif // _VLC_H_