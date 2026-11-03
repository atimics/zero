/*
 * self_edit: a learned optimizer that also edits its own weights.
 *
 * The target is a 1-8-1 tanh network fit to a random sine wave.  The editor
 * is a tiny per-parameter network: it reads one weight and that weight's
 * normalized gradient and writes the weight's next value.  Because it works
 * one parameter at a time, it can edit any weight vector, including its own.
 *
 * Phase 1 trains the editor with an outside optimizer.  Phase 2 hands the
 * editor its own meta-gradient and lets it rewrite itself, generation after
 * generation.  Two arms run from the same start on the same tasks:
 *
 *   unchecked  keeps every self-edit.
 *   checked    keeps a self-edit only if it does not raise the loss on a
 *              fixed set of validation tasks.
 *
 * Neither arm's decisions see the audit tasks, so the audit score is an
 * honest measure of both.  Meta-gradients come from random weight
 * perturbations; nothing backpropagates through the inner training run.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 8
#define TARGET_PARAMS (3 * HIDDEN + 1)
#define EDITOR_HIDDEN 6
#define EDITOR_PARAMS (4 * EDITOR_HIDDEN + 1)

#define TRAIN_POINTS 10
#define TEST_POINTS 25
#define INNER_STEPS 16

#define BATCH_TASKS 24
#define VALID_TASKS 32
#define AUDIT_TASKS 128

#define DIRECTIONS 8
#define PROBE 0.03f
#define STEP_SCALE 0.05f
#define LOSS_CAP 10.0f
#define BOOT_RATE 0.02f
#define PI_F 3.14159265f

enum { STREAM_BOOT = 1, STREAM_TRAIN, STREAM_VALID, STREAM_AUDIT, STREAM_PROBE };

typedef struct {
    uint64_t state;
} Rng;

typedef struct {
    float x[TRAIN_POINTS];
    float y[TRAIN_POINTS];
    float test_x[TEST_POINTS];
    float test_y[TEST_POINTS];
    float init[TARGET_PARAMS];
} Task;

/* An editor, or plain SGD when editor is NULL. */
typedef struct {
    const float *editor;
    float sgd_rate;
} Learner;

static void die(const char *message)
{
    fprintf(stderr, "error: %s\n", message);
    exit(EXIT_FAILURE);
}

static void rng_seed(Rng *rng, uint64_t seed)
{
    uint64_t z = seed + UINT64_C(1);
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    rng->state = z ^ (z >> 31);
    if (rng->state == 0) {
        rng->state = 1;
    }
}

/* Every random draw is keyed by (seed, stream, index) so arms share tasks. */
static void rng_keyed(Rng *rng, uint64_t seed, int stream, long index)
{
    rng_seed(rng, seed * UINT64_C(1000003) + (uint64_t)stream * UINT64_C(0x100000001b3) +
                      (uint64_t)index * UINT64_C(0x9e3779b1));
}

static uint64_t rng_next(Rng *rng)
{
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static float rng_unit(Rng *rng)
{
    return (float)((rng_next(rng) >> 40) * (1.0 / 16777216.0));
}

static float rng_symmetric(Rng *rng, float scale)
{
    return (2.0f * rng_unit(rng) - 1.0f) * scale;
}

static float clip(float value, float limit)
{
    if (value > limit) {
        return limit;
    }
    if (value < -limit) {
        return -limit;
    }
    return value;
}

static void task_make(Task *task, uint64_t seed, int stream, long index)
{
    Rng rng;
    rng_keyed(&rng, seed, stream, index);
    float amplitude = 0.5f + 1.5f * rng_unit(&rng);
    float phase = PI_F * rng_unit(&rng);
    for (int i = 0; i < TRAIN_POINTS; i++) {
        task->x[i] = rng_symmetric(&rng, PI_F);
        task->y[i] = amplitude * sinf(task->x[i] + phase);
    }
    for (int i = 0; i < TEST_POINTS; i++) {
        task->test_x[i] = rng_symmetric(&rng, PI_F);
        task->test_y[i] = amplitude * sinf(task->test_x[i] + phase);
    }
    /* Layout: w1[HIDDEN], b1[HIDDEN], w2[HIDDEN], b2. */
    for (int j = 0; j < HIDDEN; j++) {
        task->init[j] = rng_symmetric(&rng, 1.0f);
        task->init[HIDDEN + j] = rng_symmetric(&rng, 1.0f);
        task->init[2 * HIDDEN + j] = rng_symmetric(&rng, 0.5f);
    }
    task->init[3 * HIDDEN] = 0.0f;
}

/* Mean squared error of the target network; fills grad when it is non-NULL. */
static float target_loss(const float *w, const float *xs, const float *ys, int n, float *grad)
{
    const float *w1 = w;
    const float *b1 = w + HIDDEN;
    const float *w2 = w + 2 * HIDDEN;
    float loss = 0.0f;
    if (grad != NULL) {
        memset(grad, 0, TARGET_PARAMS * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        float h[HIDDEN];
        float y = w[3 * HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            h[j] = tanhf(w1[j] * xs[i] + b1[j]);
            y += w2[j] * h[j];
        }
        float err = y - ys[i];
        loss += err * err;
        if (grad != NULL) {
            float dy = 2.0f * err / (float)n;
            grad[3 * HIDDEN] += dy;
            for (int j = 0; j < HIDDEN; j++) {
                float da = dy * w2[j] * (1.0f - h[j] * h[j]);
                grad[2 * HIDDEN + j] += dy * h[j];
                grad[HIDDEN + j] += da;
                grad[j] += da * xs[i];
            }
        }
    }
    return loss / (float)n;
}

/*
 * One editor step over n weights, in place.  Editor layout: input weights
 * a[EDITOR_HIDDEN][2], hidden biases c, output weights v, output bias d.
 */
static void editor_apply(const float *theta, float *w, const float *g, int n)
{
    const float *a = theta;
    const float *c = theta + 2 * EDITOR_HIDDEN;
    const float *v = theta + 3 * EDITOR_HIDDEN;
    float d = theta[4 * EDITOR_HIDDEN];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += g[i] * g[i];
    }
    float rms = sqrtf(sum / (float)n) + 1e-8f;
    for (int i = 0; i < n; i++) {
        float f0 = clip(g[i] / rms, 5.0f);
        float f1 = clip(w[i], 5.0f);
        float out = d;
        for (int k = 0; k < EDITOR_HIDDEN; k++) {
            out += v[k] * tanhf(a[2 * k] * f0 + a[2 * k + 1] * f1 + c[k]);
        }
        w[i] += clip(STEP_SCALE * out, 1.0f);
    }
}

/* Test loss after INNER_STEPS of training on one task. */
static float task_score(const Learner *learner, const Task *task)
{
    float w[TARGET_PARAMS];
    float g[TARGET_PARAMS];
    memcpy(w, task->init, sizeof(w));
    for (int step = 0; step < INNER_STEPS; step++) {
        target_loss(w, task->x, task->y, TRAIN_POINTS, g);
        if (learner->editor != NULL) {
            editor_apply(learner->editor, w, g, TARGET_PARAMS);
        } else {
            for (int i = 0; i < TARGET_PARAMS; i++) {
                w[i] -= learner->sgd_rate * g[i];
            }
        }
    }
    float loss = target_loss(w, task->test_x, task->test_y, TEST_POINTS, NULL);
    if (!isfinite(loss) || loss > LOSS_CAP) {
        return LOSS_CAP;
    }
    return loss;
}

static float mean_score(const Learner *learner, const Task *tasks, int count)
{
    double total = 0.0;
    for (int t = 0; t < count; t++) {
        total += task_score(learner, &tasks[t]);
    }
    return (float)(total / count);
}

static float editor_score(const float *theta, const Task *tasks, int count)
{
    Learner learner = {theta, 0.0f};
    return mean_score(&learner, tasks, count);
}

/* Antithetic random-sign perturbation estimate of d(meta-loss)/d(theta). */
static void meta_gradient(const float *theta, const Task *tasks, int count, uint64_t seed,
                          long index, float *grad)
{
    Rng rng;
    rng_keyed(&rng, seed, STREAM_PROBE, index);
    memset(grad, 0, EDITOR_PARAMS * sizeof(float));
    for (int d = 0; d < DIRECTIONS; d++) {
        float sign[EDITOR_PARAMS];
        float plus[EDITOR_PARAMS];
        float minus[EDITOR_PARAMS];
        for (int i = 0; i < EDITOR_PARAMS; i++) {
            sign[i] = (rng_next(&rng) >> 63) ? 1.0f : -1.0f;
            plus[i] = theta[i] + PROBE * sign[i];
            minus[i] = theta[i] - PROBE * sign[i];
        }
        float diff = editor_score(plus, tasks, count) - editor_score(minus, tasks, count);
        for (int i = 0; i < EDITOR_PARAMS; i++) {
            grad[i] += diff / (2.0f * PROBE) * sign[i] / (float)DIRECTIONS;
        }
    }
}

static void make_tasks(Task *tasks, int count, uint64_t seed, int stream, long first)
{
    for (int t = 0; t < count; t++) {
        task_make(&tasks[t], seed, stream, first + t);
    }
}

static float norm(const float *x, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}

static long parse_long(const char *text, const char *option)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0) {
        fprintf(stderr, "error: bad value for %s: %s\n", option, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--seed N] [--boot N] [--generations N]\n"
            "writes one CSV row per generation to stdout\n",
            program);
}

int main(int argc, char **argv)
{
    uint64_t seed = 1;
    long boot = 300;
    long generations = 300;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc && strcmp(argv[i], "--seed") == 0) {
            seed = (uint64_t)parse_long(argv[++i], "--seed");
        } else if (i + 1 < argc && strcmp(argv[i], "--boot") == 0) {
            boot = parse_long(argv[++i], "--boot");
        } else if (i + 1 < argc && strcmp(argv[i], "--generations") == 0) {
            generations = parse_long(argv[++i], "--generations");
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    static Task batch[BATCH_TASKS];
    static Task valid[VALID_TASKS];
    static Task audit[AUDIT_TASKS];
    make_tasks(valid, VALID_TASKS, seed, STREAM_VALID, 0);
    make_tasks(audit, AUDIT_TASKS, seed, STREAM_AUDIT, 0);

    /* Reference points: no training at all, and the best of a few SGD rates. */
    Learner idle = {NULL, 0.0f};
    float idle_audit = mean_score(&idle, audit, AUDIT_TASKS);
    float best_sgd = LOSS_CAP;
    float best_rate = 0.0f;
    const float rates[] = {0.01f, 0.03f, 0.1f, 0.3f, 1.0f};
    for (size_t r = 0; r < sizeof(rates) / sizeof(rates[0]); r++) {
        Learner sgd = {NULL, rates[r]};
        float score = mean_score(&sgd, audit, AUDIT_TASKS);
        if (score < best_sgd) {
            best_sgd = score;
            best_rate = rates[r];
        }
    }

    /* Phase 1: an outside optimizer (normalized gradient descent) trains the editor. */
    float theta[EDITOR_PARAMS];
    float grad[EDITOR_PARAMS];
    Rng rng;
    rng_keyed(&rng, seed, STREAM_BOOT, -1);
    for (int i = 0; i < EDITOR_PARAMS; i++) {
        theta[i] = rng_symmetric(&rng, 0.5f);
    }
    float random_audit = editor_score(theta, audit, AUDIT_TASKS);
    for (long b = 0; b < boot; b++) {
        make_tasks(batch, BATCH_TASKS, seed, STREAM_BOOT, b * BATCH_TASKS);
        meta_gradient(theta, batch, BATCH_TASKS, seed, -1 - b, grad);
        float rms = norm(grad, EDITOR_PARAMS) / sqrtf((float)EDITOR_PARAMS) + 1e-8f;
        for (int i = 0; i < EDITOR_PARAMS; i++) {
            theta[i] -= BOOT_RATE * grad[i] / rms;
        }
    }

    printf("# seed=%llu boot=%ld generations=%ld editor_params=%d target_params=%d\n",
           (unsigned long long)seed, boot, generations, EDITOR_PARAMS, TARGET_PARAMS);
    printf("# audit loss: no_training=%.4f random_editor=%.4f best_sgd=%.4f (rate %.2f)\n",
           idle_audit, random_audit, best_sgd, best_rate);
    printf("generation,unchecked_audit,checked_audit,checked_kept,unchecked_norm,checked_norm\n");

    /* Phase 2: each arm's editor edits itself with its own meta-gradient. */
    float arm[2][EDITOR_PARAMS];
    memcpy(arm[0], theta, sizeof(theta));
    memcpy(arm[1], theta, sizeof(theta));
    float checked_valid = editor_score(arm[1], valid, VALID_TASKS);
    long kept = 0;
    float start = editor_score(theta, audit, AUDIT_TASKS);
    printf("0,%.5f,%.5f,0,%.4f,%.4f\n", start, start, norm(theta, EDITOR_PARAMS),
           norm(theta, EDITOR_PARAMS));
    for (long gen = 1; gen <= generations; gen++) {
        make_tasks(batch, BATCH_TASKS, seed, STREAM_TRAIN, gen * BATCH_TASKS);
        for (int a = 0; a < 2; a++) {
            float proposal[EDITOR_PARAMS];
            meta_gradient(arm[a], batch, BATCH_TASKS, seed, gen, grad);
            memcpy(proposal, arm[a], sizeof(proposal));
            editor_apply(arm[a], proposal, grad, EDITOR_PARAMS);
            if (a == 0) {
                memcpy(arm[0], proposal, sizeof(proposal));
                continue;
            }
            float score = editor_score(proposal, valid, VALID_TASKS);
            if (score <= checked_valid) {
                memcpy(arm[1], proposal, sizeof(proposal));
                checked_valid = score;
                kept++;
            }
        }
        printf("%ld,%.5f,%.5f,%ld,%.4f,%.4f\n", gen, editor_score(arm[0], audit, AUDIT_TASKS),
               editor_score(arm[1], audit, AUDIT_TASKS), kept, norm(arm[0], EDITOR_PARAMS),
               norm(arm[1], EDITOR_PARAMS));
        fflush(stdout);
    }
    if (ferror(stdout)) {
        die("could not write output");
    }
    return EXIT_SUCCESS;
}
