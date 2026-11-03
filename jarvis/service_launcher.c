/*
 * JARVIS.app launcher.
 *
 * macOS grants microphone / automation permissions to the "responsible" app. A bare Python
 * process started in the background often gets silently denied, so JARVIS runs inside a tiny
 * app bundle whose executable is this launcher. It *spawns* Python (rather than exec-ing it)
 * so the permission prompts say "JARVIS" and the grants stick to this app.
 *
 * Built by `uv run jarvis-service install`; JARVIS_DIR and JARVIS_PYTHON are passed with -D.
 */
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(void) {
    if (chdir(JARVIS_DIR) != 0) {
        perror("jarvis launcher: chdir");
        return 1;
    }
    setenv("PYTHONUNBUFFERED", "1", 1);  /* so the service log updates live */
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);

    char *argv[] = {JARVIS_PYTHON, "-m", "jarvis", NULL};
    int err = posix_spawn(&child, JARVIS_PYTHON, NULL, NULL, argv, environ);
    if (err != 0) {
        fprintf(stderr, "jarvis launcher: could not start %s (error %d)\n", JARVIS_PYTHON, err);
        return 1;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        /* interrupted by a forwarded signal: keep waiting for Python to exit */
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128 + WTERMSIG(status);
}
