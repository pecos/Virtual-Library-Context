/**
 * @brief a heck to enable std::thread when pthread shim is used
 * 
 * In glibc++, it will check if pthread exists by checking symbol "__pthread_key_create".
 * And it is not possible to define that symbol in the pthread shim since
 * pthread constructor will call __pthread_key_create which conflict to shim symbol loading.
 * 
 * With LD_PRELOAD=pthread_patch.so, we could ensure this symbol will appear before glibc++ is loaded,
 * so `__gthread_active_p` will return true and pass the checking the exist of pthread.
 * 
 * @ref https://stackoverflow.com/a/77482634/18625467
*/
int
__pthread_key_create (unsigned int *key, void (*destr) (void *))
{
    return 11;  // EPERM
};
