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
#include <sys/syscall.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <seccomp.h>
#include <sched.h>
#include <unordered_map>
#include <cassert>

#ifdef NDEBUG
#define DEBUG(...) 
#else
#define DEBUG(...) ({\
            printf("[DEBUG] ");\
            printf(__VA_ARGS__);\
            printf("\n");\
           })
#endif

namespace VLC {

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
    void initialize() {
        pid_t pid = fork();

        if (pid == -1) {
            std::cerr << "VLC: unable to start VLC manager, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        } else if (pid == 0) {
            // this is application process (child)
            
            // set a BPF filter on specific syscall for ptrace
            configure_seccomp_bpf();
            raise(SIGSTOP); // stop and let monitor process know

            // return to the application's main()
            std::cout << "VLC: application start." << std::endl;
            return;
        } else {  
            // this is manager process (parent)
            std::cout << "VLC: manager start." << std::endl;
            application_pid = pid;
            application_child_states[pid] = ChildState::RUNNING; // itself is also in the list
            determine_resouces();

            // wait on application seccomp bpf configuration
            int status;
            if (waitpid(-1, &status, __WALL) == -1) {
                std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // let application terminate along with monitor process
            if (ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_EXITKILL | PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP) == -1) {
                std::cerr << "VLC: unable to config tracer 1." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (ptrace(PTRACE_CONT, application_pid, 0, 0) == -1) {
                std::cerr << "VLC: unable to config tracer 5." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // start monitoring the application
            intercept_syscall();

            // exit when the application finished
            std::exit(EXIT_SUCCESS);  
        }
    }

private:
    enum ChildState {NEW_STOPPED, NEW_FORKED, RUNNING};

    cpu_set_t system_cpu_set;
    cpu_set_t virtual_cpu_set;
    bool virtual_cpu_set_is_set = false;
    pid_t application_pid = 0;
    std::unordered_map<pid_t, ChildState> application_child_states;  // map of child to states of the application process (including itself)
    int forge_sched_getaffinity_count = 0;

    /**
     * Configure a BPF filter for ptrace + seccomp.
     * 
     * Will mark the caller process as tracee.
     * This allows the tracer only stop on specific syscall
     * that related to resouces query rather than all of them.
     * So the overhead of interception could be minimized.
    */
   void configure_seccomp_bpf() {
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
            pid_t child_waited = waitpid(-1, &status, __WALL);
            if (child_waited == -1) {
                std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            
            DEBUG("VLC: stop application, pid=%d.", child_waited);

            // if haven't seen this pid before
            if (application_child_states.find(child_waited) == application_child_states.end()) {
                // if this is a SIGSTOP event (a new child is stopped at begin)
                if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
                    DEBUG("VLC: a child is stopped after created, pid=%d", child_waited);
                    application_child_states[child_waited] = ChildState::NEW_STOPPED;
                    continue;
                } else {
                    std::cerr << "VLC: found unknown process/thread which is not traced, pid=" << child_waited << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }

            // stopped on seccomp signal
            if (status >> 8 == (SIGTRAP | (PTRACE_EVENT_SECCOMP << 8))) {
                // we alreay at the syscall entry point,
                // so continue until syscall exit
                if (ptrace(PTRACE_SYSCALL, child_waited, 0, 0) == -1)  {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (waitpid(child_waited, &status, 0) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                
                // retrive syscall arguments
                user_regs_struct regs;
                if (ptrace(PTRACE_GETREGS, child_waited, 0, &regs) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                long syscall = regs.orig_rax;

                if (syscall == SYS_sched_getaffinity) {  // capture sys_sched_getaffinity
                    if (forge_sched_getaffinity_count < 2) {
                        forge_sched_getaffinity_count++;
                    } else {
                        forge_sched_getaffinity(regs.rdx);
                    }
                }
            } else if ((status >> 8 == (SIGTRAP | (PTRACE_EVENT_CLONE << 8))) ||
                (status >> 8 == (SIGTRAP | (PTRACE_EVENT_FORK << 8)))) {
                // child called a clone (create a new child)
                // need to trace the new child
                pid_t new_child;
                if (ptrace(PTRACE_GETEVENTMSG, child_waited, 0, &new_child) == -1) {
                    std::cerr << "VLC: unable to retrive new child pid, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                DEBUG("VLC: a new child process/thread is created, pid=%d", new_child);

                // if the pid is first seen
                if (application_child_states.find(new_child) == application_child_states.end()) {
                    application_child_states[new_child] = ChildState::NEW_FORKED;
                } else {
                    // if the child stop event is already caputured
                    assert(application_child_states[new_child] == ChildState::NEW_STOPPED && "Child state invalid");
                
                    // trace and let child continue
                    if (ptrace(PTRACE_SETOPTIONS, new_child, 0, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
                        std::cerr << "VLC: unable to config tracer 6, pid=" << new_child << ", " << strerror(errno) << std::endl;
                        std::exit(EXIT_FAILURE);
                    }

                    DEBUG("VLC: child is traced and resumed, pid=%d", new_child);

                    application_child_states[new_child] == ChildState::RUNNING;
                    if (ptrace(PTRACE_CONT, new_child, 0, 0) == -1) {
                        std::cerr << "VLC: unable to config tracer 8." << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            } else if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
                // a child is stopped after created, and fork event already recived
                assert(application_child_states[child_waited] == ChildState::NEW_FORKED && "Child state is invalid");
                
                // trace and let child continue
                if (ptrace(PTRACE_SETOPTIONS, child_waited, 0, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
                    std::cerr << "VLC: unable to config tracer 6, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                DEBUG("VLC: child is traced and resumed, pid=%d", child_waited);

                // CONT will be sent after the if block
                application_child_states[child_waited] == ChildState::RUNNING;
            } else if (WIFEXITED(status) || WIFSIGNALED(status)) {
                // child has exited or terminated
                application_child_states.erase(child_waited);
                DEBUG("VLC: a child process exit, pid=%d", child_waited);

                if (application_child_states.size() == 0) {
                    std::cout << "VLC: manager exit since application has exited (NOT A ERROR)." << std::endl;
                    break;
                } 

                // child already exist, skip the rest 
                continue; 
            } else {
                std::cerr << "VLC: stop on an unknown event " << status << ", pid=" << child_waited << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // continue until next seccomp signal
            if (ptrace(PTRACE_CONT, child_waited, 0, 0) == -1) {
                std::cerr << "VLC: unable to continue the application, pid=" << child_waited << ", " << strerror(errno) << std::endl;
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
        
        // copy the virtual cpu set into application's memory
        iovec local_iov[1];
        local_iov[0].iov_base = &virtual_cpu_set;
        local_iov[0].iov_len = sizeof(cpu_set_t);

        iovec remote_iov[1];
        remote_iov[0].iov_base = (void *) user_mask_ptr;
        remote_iov[0].iov_len = sizeof(cpu_set_t);

        if (process_vm_writev(application_pid, local_iov, 1, remote_iov, 1, 0) == -1) {
            std::cerr << "VLC: unable to modify sched_getaffinity(), " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        DEBUG("VLC: sched_getaffinity() is modifed.");
    }
};
    
// void enable() {
//     std::thread::id this_id = std::this_thread::get_id();
// }

// void register_thread(pthread_t thread, int id);

}
#endif // _VLC_H_