// Measure full executable lifetime without Python timeout-wait polling latency.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if(argc<3) return 2;
    auto start=std::chrono::steady_clock::now();
    pid_t child=fork();
    if(child<0) return 3;
    if(child==0) {execv(argv[2],argv+2); _exit(127);}
    int status; struct rusage usage{};
    while(wait4(child,&status,0,&usage)<0) {
        if(errno!=EINTR) return 4;
    }
    auto end=std::chrono::steady_clock::now();
    std::ofstream output(argv[1]);
    output<<std::setprecision(17)<<"{\"seconds\":"
          <<std::chrono::duration<double>(end-start).count()
          <<",\"user_seconds\":"<<(usage.ru_utime.tv_sec+usage.ru_utime.tv_usec*1e-6)
          <<",\"system_seconds\":"<<(usage.ru_stime.tv_sec+usage.ru_stime.tv_usec*1e-6)
          <<",\"max_rss_bytes\":"<<(usage.ru_maxrss*1024LL)
          <<",\"pid\":"<<child<<"}\n";
    return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
}
