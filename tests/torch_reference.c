#define main zero_cli_main
#include "../literary_lm.c"
#undef main

int main(int argc, char **argv)
{
    Config cfg = preset_config("literary");
    Model model;
    Rng rng;
    Token input[16], target[16];
    FILE *file;
    int i;
    if (argc != 3) return 2;
    rng_seed(&rng, 7);
    if (strcmp(argv[1], "initialize") == 0) {
        model_create(&model, cfg, &rng);
        checkpoint_save(argv[2], &model, 0, &rng);
        model_destroy(&model);
        return 0;
    }
    cfg.context = 16; cfg.dim = 16; cfg.heads = 2; cfg.layers = 2; cfg.ff = 32;
    model_create(&model, cfg, &rng);
    checkpoint_save(argv[2], &model, 0, &rng);
    for (i = 0; i < 16; ++i) { input[i] = 32 + i; target[i] = 33 + i; }
    model_zero_grad(&model);
    printf("loss=%.9f\n", model_forward(&model, input, target, 0, NULL));
    file = fopen("reference.bin", "wb");
    if (!file) return 3;
    fwrite(model.probs, sizeof(float), 16 * cfg.vocab, file);
    model_backward(&model, input, target);
    for (i = 0; i < model.parameter_count; ++i)
        fwrite(model.parameters[i]->g, sizeof(float), model.parameters[i]->count, file);
    fclose(file);
    optimizer_update(&model, 1, 0.0003f, 0.01f, 1.0f, 1.0f);
    checkpoint_save("updated.ckpt", &model, 1, &rng);
    model_destroy(&model);
    return 0;
}
