#ifndef _VLC_RUNTIME_H_
#define _VLC_RUNTIME_H_

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

#include "VLC/info.h"
#include "VLC/resource.h"
#include "VLC/communication.h"

// this variable will has same address on Manager and Application processes
static const char FORGED_CPU_FOLDER[] = "/home/yyan/forge-cpu/zemaitis/full/cpu";
static const char FORGED_CPU_ONLINE_FILE[] = "/home/yyan/forge-cpu/zemaitis/full/cpu/online";
static const char FORGED_CPU_INFO_FILE[] = "/home/yyan/forge-cpu/zemaitis/full/cpuinfo";
// static char FORGED_MEM_INFO_FILE_BUFFER[] = "                                                a";

// static const char FORGED_CPU_FOLDER_1[] = "/home/yyan/vlc/zemaitis/4/cpu";
// static const char FORGED_CPU_INFO_FILE_1[] = "/home/yyan/vlc/zemaitis/4/cpuinfo";
// static const char FORGED_CPU_FOLDER_2[] = "/home/yyan/vlc/zemaitis/full/cpu";
// static const char FORGED_CPU_INFO_FILE_2[] = "/home/yyan/vlc/zemaitis/full/cpuinfo";

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
        // prepare shared memory for future communication
        Internal::VLC_SHARED_MEM = Internal::SharedMem(sizeof(VLC::Context));
        Internal::MONITOR_PID = getpid();

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

            // register signal hanlder for communication
            signal(SIGUSR1, Internal::monitor_sig_hanlder);

            // wait on application seccomp bpf configuration
            int status;
            if (waitpid(-1, &status, __WALL) == -1) {
                VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
            }

            // let application terminate along with monitor process
            if (ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_EXITKILL | PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP) == -1) {
                VLC_DIE("VLC: unable to config tracer.");
            }

            std::cout << "VLC: resume application, pid=" << application_pid << std::endl;
            if (ptrace(PTRACE_CONT, application_pid, 0, 0) == -1) {
                VLC_DIE("VLC: unable to resume application.");
            }

            // start monitoring the application
            intercept_syscall();

            // exit when the application finished
            std::exit(EXIT_SUCCESS);  
        }
    }

private:
    enum ChildState {NEW_STOPPED, NEW_FORKED, RUNNING};

    // VLC::Internal::Resource resource;

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
            VLC_DIE("VLC: unable to promise NO_NEW_PRIVS");
        }

        // set seccomp with the above BFP program
        if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) == -1) {
            VLC_DIE("VLC: untable to set seccomp filter, %s", strerror(errno));
        }
   }

    /**
     * Query this system's resouce limits,
     * including cpu set
    */
    void determine_resouces() {
        // cpu set
        if (sched_getaffinity(0, sizeof(cpu_set_t), &Internal::system_cpu_set) == -1) {
            VLC_DIE("VLC: unable to determine cpu set, %s", strerror(errno));
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
                VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
            }
            
            VLC_DEBUG("VLC: stop application, pid=%d.", child_waited);

            // if haven't seen this pid before
            if (application_child_states.find(child_waited) == application_child_states.end()) {
                // if this is a SIGSTOP event (a new child is stopped at begin)
                if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
                    VLC_DEBUG("VLC: a child is stopped after created, pid=%d", child_waited);
                    application_child_states[child_waited] = ChildState::NEW_STOPPED;
                    continue;
                } else {
                    VLC_DIE("VLC: found unknown process/thread which is not traced, pid=%d", child_waited);
                }
            }

            // stopped on seccomp signal
            if (status >> 8 == (SIGTRAP | (PTRACE_EVENT_SECCOMP << 8))) {
                /*** Phase 1: entering syscall **/

                // retrive syscall arguments
                user_regs_struct regs;
                if (ptrace(PTRACE_GETREGS, child_waited, NULL, &regs) == -1) {
                    VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
                }
                long syscall = regs.orig_rax;
     
                if (syscall == SYS_openat) {  // capture openat()
                    forge_openat(child_waited, &regs.rsi);
                    ptrace(PTRACE_SETREGS, child_waited, NULL, &regs);
                }

                /*** Phase 2: waiting syscall execution **/

                // continue until syscall exit
                if (ptrace(PTRACE_SYSCALL, child_waited, NULL, NULL) == -1)  {
                    VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
                }
                if (waitpid(child_waited, &status, 0) == -1) {
                    VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
                }

                /*** Phase 3: leaving syscall **/
                
                // retrive syscall arguments
                if (ptrace(PTRACE_GETREGS, child_waited, NULL, &regs) == -1) {
                    VLC_DIE("VLC: unable to intercept syscall, %s", strerror(errno));
                }
                assert(regs.orig_rax == syscall);  // orig_rax should not change

                // for x86 ABI
                // check https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
                if (syscall == SYS_sched_getaffinity) {  // capture sched_getaffinity()
                    forge_sched_getaffinity(child_waited, regs.rsi, regs.rdx);
                }
            } else if ((status >> 8 == (SIGTRAP | (PTRACE_EVENT_CLONE << 8))) ||
                (status >> 8 == (SIGTRAP | (PTRACE_EVENT_FORK << 8)))) {
                // child called a clone (create a new child)
                // need to trace the new child
                pid_t new_child;
                if (ptrace(PTRACE_GETEVENTMSG, child_waited, NULL, &new_child) == -1) {
                    VLC_DIE("VLC: unable to retrive new child pid, %s", strerror(errno));
                }

                VLC_DEBUG("VLC: a new child process/thread is created, pid=%d, parent=%d", new_child, child_waited);

                // put the new child to the same VLC its parent in
                VLC::Internal::pid_to_vlc_id[new_child] = VLC::Internal::pid_to_vlc_id[child_waited];
                VLC_DEBUG("VLC: new_child %d is in vlc %d", new_child, VLC::Internal::pid_to_vlc_id[new_child]);

                // if the pid is first seen
                if (application_child_states.find(new_child) == application_child_states.end()) {
                    application_child_states[new_child] = ChildState::NEW_FORKED;
                } else {
                    // if the child stop event is already caputured
                    assert(application_child_states[new_child] == ChildState::NEW_STOPPED && "Child state invalid");
                    
                    // set affinity on new child if it is on a VLC
                    if (VLC::Internal::pid_to_vlc_id[new_child] != 0) {
                        Internal::create_virtual_affinity(new_child);
                        Internal::enfore_virtual_affinity(new_child);
                    }
                    
                    // trace and let child continue
                    if (ptrace(PTRACE_SETOPTIONS, new_child, NULL, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
                        VLC_DIE("VLC: unable to config tracer on child");
                    }

                    VLC_DEBUG("VLC: child is traced and resumed, pid=%d", new_child);

                    application_child_states[new_child] = ChildState::RUNNING;
                    if (ptrace(PTRACE_CONT, new_child, NULL, NULL) == -1) {
                        VLC_DIE("VLC: unable to config tracer");
                    }
                }
            } else if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
                // a child is stopped after created, and fork event already recived
                assert(application_child_states[child_waited] == ChildState::NEW_FORKED && "Child state is invalid");
                
                // set affinity on new child if it is on a VLC
                if (VLC::Internal::pid_to_vlc_id[child_waited] != 0) {
                    Internal::create_virtual_affinity(child_waited);
                    Internal::enfore_virtual_affinity(child_waited);
                }

                // trace and let child continue
                if (ptrace(PTRACE_SETOPTIONS, child_waited, NULL, PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACESECCOMP | PTRACE_O_EXITKILL) == -1) {
                    VLC_DIE("VLC: unable to config tracer");
                }

                VLC_DEBUG("VLC: child is traced and resumed, pid=%d", child_waited);

                // CONT will be sent after the if block
                application_child_states[child_waited] = ChildState::RUNNING;
            } else if (WIFEXITED(status) || WIFSIGNALED(status)) {
                if (WIFEXITED(status)) {
                    VLC_DEBUG("VLC: child exit with status %d, pid=%d", WEXITSTATUS(status), child_waited);
                }

                if (WIFSIGNALED(status)) {
                    VLC_DEBUG("VLC: child terminated by signal %d (%s) %s, pid=%d", WTERMSIG(status), strsignal(WTERMSIG(status)), 
                        (WCOREDUMP(status) ? " (core dumped)" : ""), child_waited);
                }

                // child has exited or terminated
                application_child_states.erase(child_waited);
                
                if (application_child_states.size() == 0) {
                    std::cout << "VLC: manager exit since application has exited (NOT A ERROR)." << std::endl;
                    break;
                } 

                // child already exist, skip the rest 
                continue; 
            } else if (status >> 8 == SIGUSR1) {
                // this is a signal for communication between monitor and application
                VLC_DEBUG("VLC: recive SIGUSR1, continued, pid=%d", child_waited);
            } else {
               if (WIFSTOPPED(status)) {
                    VLC_DIE("VLC: stopped by signal %d (%s), pid=%d", WSTOPSIG(status), strsignal(WSTOPSIG(status)), child_waited);
                }
                else if (WIFCONTINUED(status)) {
                    VLC_DEBUG("VLC: recive SIGCONT, continued, pid=%d", child_waited);
                } else {
                    VLC_DIE("VLC: stop on an unknown status %d, pid=%d", status, child_waited);
                }
            }

            // continue until next seccomp signal
            if (ptrace(PTRACE_CONT, child_waited, NULL, NULL) == -1) {
                VLC_DIE("VLC: unable to continue the application, pid=%d, %s", child_waited, strerror(errno));
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
        // make a virtual cpu affinity
        Internal::create_virtual_affinity(pid);
        
        // copy the virtual cpu set into application's memory
        iovec local_iov[1];
        local_iov[0].iov_base = &( Internal::virtual_cpu_sets[pid]);
        local_iov[0].iov_len = len;

        iovec remote_iov[1];
        remote_iov[0].iov_base = (void *) user_mask_ptr;
        remote_iov[0].iov_len = len;

        if (process_vm_writev(application_pid, local_iov, 1, remote_iov, 1, 0) == -1) {
            VLC_DIE("VLC: unable to modify sched_getaffinity(), %s", strerror(errno));
        }

        VLC_DEBUG("VLC: sched_getaffinity() is modifed.");
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
        VLC_DEBUG("VLC: application try to open %s", filename_str);
        
        // check if the path is cpu resouce file
        // modify the pointer value to a pre defined str in the application space
        if (strcmp(filename_str, "/sys/devices/system/cpu") == 0) {
            *filename_ptr = (unsigned long long) FORGED_CPU_FOLDER;
            VLC_DEBUG("VLC: open %s", FORGED_CPU_FOLDER);
        } else if (strcmp(filename_str, "/sys/devices/system/cpu/online") == 0) {
            *filename_ptr = (unsigned long long) FORGED_CPU_ONLINE_FILE;
            VLC_DEBUG("VLC: open %s", FORGED_CPU_ONLINE_FILE);
        } else if (strcmp(filename_str, "/proc/cpuinfo") == 0) {
            *filename_ptr = (unsigned long long) FORGED_CPU_INFO_FILE;
            VLC_DEBUG("VLC: open %s", FORGED_CPU_INFO_FILE);
        } 
        // else if (strcmp(filename_str, "/proc/meminfo") == 0) {
        //     resource.set_avaliable_mem(0, 8);
        //     resource.generate_mem_info_file(0, FORGED_MEM_INFO_FILE_BUFFER);
        //     *filename_ptr = (unsigned long long) FORGED_MEM_INFO_FILE_BUFFER;
        // } 
        else {
            free(filename_str);
            return;
        }

        free(filename_str);
        VLC_DEBUG("VLC: openat() is modifed.");
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
            VLC_DIE("VLC: unable to malloc, %s", strerror(errno));
        }

        while (true) {
            if (read + (int) sizeof(unsigned long long) > allocated) {
                allocated *= 2;
                str = (char *) realloc(str, allocated);
                if (!str) {
                    VLC_DIE("VLC: unable to malloc, %s", strerror(errno));
                }
            }
            // clear errno before check it
            errno = 0;
            long ret = ptrace(PTRACE_PEEKDATA, pid, addr + read, NULL);
            if(errno != 0) {
                VLC_DIE("VLC: unable to peak string from application memory, %s", strerror(errno));
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
#endif // _VLC_RUNTIME_H_