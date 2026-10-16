// Private full-child monotonic wait4 clock; PID file permits external observation.
#include <chrono>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc,char** argv) {
  if(argc<4) return 2;
  auto start=std::chrono::steady_clock::now();
  pid_t child=fork();
  if(child<0) return 3;
  if(child==0) {
    rlimit lim{8ULL<<30,8ULL<<30};
    if(setrlimit(RLIMIT_AS,&lim)!=0) _exit(125);
    execv(argv[3],argv+3); _exit(127);
  }
  {std::ofstream p(argv[2]); p<<child<<"\n";}
  int status; rusage usage{};
  while(wait4(child,&status,0,&usage)<0) if(errno!=EINTR) return 4;
  auto end=std::chrono::steady_clock::now();
  std::ofstream out(argv[1]);
  out<<std::setprecision(17)<<"{\"seconds\":"
     <<std::chrono::duration<double>(end-start).count()
     <<",\"user_seconds\":"<<(usage.ru_utime.tv_sec+usage.ru_utime.tv_usec*1e-6)
     <<",\"system_seconds\":"<<(usage.ru_stime.tv_sec+usage.ru_stime.tv_usec*1e-6)
     <<",\"max_rss_bytes\":"<<usage.ru_maxrss*1024LL
     <<",\"pid\":"<<child<<",\"exit_code\":"
     <<(WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status))<<"}\n";
  return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
}
