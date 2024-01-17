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

// this variable will has same address on Manager and Application processes
static const char FORGED_CPU_FOLDER[] = "/home/yyan/cpu";
static const char FORGED_CPU_ONLINE_FILE[] = "/home/yyan/cpu/online";

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
    std::unordered_map<pid_t, cpu_set_t> virtual_cpu_sets;
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
            // TODO: add more syscall here
            // if it is open(), return TRACE
            // BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, SYS_open, 3, 0),
            // if it is openat(), return TRACE
            BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, SYS_openat, 2, 0),
            // if it is sched_getaffinity(), return TRACE
            BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, SYS_sched_getaffinity, 1, 0),
            // else, continue the syscall without tracing
            BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_ALLOW),
            BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_TRACE),
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
                /*** Phase 1: entering syscall **/

                // retrive syscall arguments
                user_regs_struct regs;
                if (ptrace(PTRACE_GETREGS, child_waited, NULL, &regs) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                long syscall = regs.orig_rax;
     
                if (syscall == SYS_openat) {  // capture openat()
                    forge_openat(child_waited, &regs.rsi);
                    ptrace(PTRACE_SETREGS, child_waited, NULL, &regs);
                }

                /*** Phase 2: waiting syscall execution **/

                // continue until syscall exit
                if (ptrace(PTRACE_SYSCALL, child_waited, NULL, NULL) == -1)  {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (waitpid(child_waited, &status, 0) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                /*** Phase 3: leaving syscall **/
                
                // retrive syscall arguments
                if (ptrace(PTRACE_GETREGS, child_waited, NULL, &regs) == -1) {
                    std::cerr << "VLC: unable to intercept syscall, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                assert(regs.orig_rax == syscall);  // orig_rax should not change

                // for x86 ABI
                // check https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
                if (syscall == SYS_sched_getaffinity) {  // capture sched_getaffinity()
                    forge_sched_getaffinity(child_waited, regs.rsi, regs.rdx);

                    // enforce the affinity we virtulized
                    if (sched_setaffinity(child_waited, regs.rsi, &virtual_cpu_sets[child_waited]) == -1) {
                        std::cerr << "VLC: unable to set cpu set, " << strerror(errno) << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            } else if ((status >> 8 == (SIGTRAP | (PTRACE_EVENT_CLONE << 8))) ||
                (status >> 8 == (SIGTRAP | (PTRACE_EVENT_FORK << 8)))) {
                // child called a clone (create a new child)
                // need to trace the new child
                pid_t new_child;
                if (ptrace(PTRACE_GETEVENTMSG, child_waited, NULL, &new_child) == -1) {
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
                    if (ptrace(PTRACE_SETOPTIONS, new_child, NULL, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
                        std::cerr << "VLC: unable to config tracer 6, pid=" << new_child << ", " << strerror(errno) << std::endl;
                        std::exit(EXIT_FAILURE);
                    }

                    DEBUG("VLC: child is traced and resumed, pid=%d", new_child);

                    application_child_states[new_child] == ChildState::RUNNING;
                    if (ptrace(PTRACE_CONT, new_child, NULL, NULL) == -1) {
                        std::cerr << "VLC: unable to config tracer 8." << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            } else if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
                // a child is stopped after created, and fork event already recived
                assert(application_child_states[child_waited] == ChildState::NEW_FORKED && "Child state is invalid");
                
                // trace and let child continue
                if (ptrace(PTRACE_SETOPTIONS, child_waited, NULL, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
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
            if (ptrace(PTRACE_CONT, child_waited, NULL, NULL) == -1) {
                std::cerr << "VLC: unable to continue the application, pid=" << child_waited << ", " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * modify the result of sched_getaffinity().
     * will replace original content in `user_mask_ptr`
     * with a virtualized cpu affinity.
     * 
     * @param len the number of bytes for user_mask_ptr
     * @param user_mask_ptr a remote address of application's cpu set buffer
     * Its content will be modified.
    */
    void forge_sched_getaffinity(pid_t pid, unsigned int len, unsigned long long user_mask_ptr) {
        // TODO: add policy here
        // make a virtual cpu affinity
        DEBUG("VLC: len=%d.", len);

        if (virtual_cpu_sets.find(pid) == virtual_cpu_sets.end()) {
            cpu_set_t virtual_cpu_set = system_cpu_set;
            
            // if (virtual_cpu_sets.size() % 2 == 0) {
            //     for (int i = 0; i < 48; i++) {
            //         if (i % 2 == 0) {
            //             CPU_CLR(i, &virtual_cpu_set);
            //         }
            //     }
            // } else {
            //     for (int i = 0; i < 48; i++) {
            //         if (i % 2 != 0) {
            //             CPU_CLR(i, &virtual_cpu_set);
            //         }
            //     }
            // }
            if (virtual_cpu_sets.size() % 2 == 0) {
                for (int i = 0; i < 24; i++) {
                    CPU_CLR(i, &virtual_cpu_set);
                }
            } else {
                for (int i = 24; i < 48; i++) {
                    CPU_CLR(i, &virtual_cpu_set);
                }
            }

            virtual_cpu_sets[pid] = std::move(virtual_cpu_set);
        }
        
        // copy the virtual cpu set into application's memory
        iovec local_iov[1];
        local_iov[0].iov_base = &(virtual_cpu_sets[pid]);
        local_iov[0].iov_len = len;

        iovec remote_iov[1];
        remote_iov[0].iov_base = (void *) user_mask_ptr;
        remote_iov[0].iov_len = len;

        if (process_vm_writev(application_pid, local_iov, 1, remote_iov, 1, 0) == -1) {
            std::cerr << "VLC: unable to modify sched_getaffinity(), " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        DEBUG("VLC: sched_getaffinity() is modifed.");
    }

    /**
     * modify the result of openat().
     * 
     * if it opening a cpu resouce file
     * will replace filename so it will open a forged file instead.
     * 
     * @param user_mask_ptr a remote address of filename
     * Its content will be modified.
    */
    void forge_openat(pid_t pid, unsigned long long *filename_ptr) {
        char *filename_str = ptrace_peak_string(pid, *filename_ptr);
        DEBUG("VLC: application try to open %s", filename_str);
        
        // check if the path is cpu resouce file
        // modify the pointer value to a pre defined str in the application space
        if (strcmp(filename_str, "/sys/devices/system/cpu") == 0) {
            *filename_ptr = (unsigned long long) FORGED_CPU_FOLDER;
        } else if (strcmp(filename_str, "/sys/devices/system/cpu/online") == 0) {
            *filename_ptr = (unsigned long long) FORGED_CPU_ONLINE_FILE;
        } else {
            return;
        }

        free(filename_str);
        DEBUG("VLC: openat() is modifed.");
    }

    /**
     * read string from remote address space
     * 
     * @param pid id of target process
     * @param addr remote address in target process's address space
     * 
     * @return the pointer to the string
     * 
     * @note caller is responsible to free the returned string
    */
    char *ptrace_peak_string(pid_t pid, unsigned long long addr) {
        int allocated = 128, read = 0;
        char *str = (char *) malloc(allocated);
        if (!str) {
            std::cerr << "VLC: unable to malloc, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        while (true) {
            if (read + sizeof(unsigned long long) > allocated) {
                allocated *= 2;
                str = (char *) realloc(str, allocated);
                if (!str) {
                    std::cerr << "VLC: unable to malloc, " << strerror(errno) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
            // clear errno before check it
            errno = 0;
            long ret = ptrace(PTRACE_PEEKDATA, pid, addr + read, NULL);
            if(errno != 0) {
                std::cerr << "VLC: unable to peak string from application memory, " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            memcpy(str + read, &ret, sizeof(ret));

            // check termination of string
            // if found 0, the string is terminated
            if (memchr(&ret, 0, sizeof(ret))) break;

            read += sizeof(ret);
        }
        return str;
    }
};
    
// void enable() {
//     std::thread::id this_id = std::this_thread::get_id();
// }

// void register_thread(pthread_t thread, int id);

}
#endif // _VLC_H_