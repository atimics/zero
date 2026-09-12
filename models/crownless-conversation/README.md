# Crownless conversation core

The **4,935,937-parameter** conversation model reads recent speech alongside
the next avatar's held account. Its saved file is **5,044,766 bytes**. It uses
the tokenizer in `models/crownless-core-v2/tokenizer.json` and the
[MIT model license](../LICENSE).

Run `scripts/chat_crownless.py` to alternate two avatars. The runner appends
each generated line to the following avatar's input. See the
[report and reproduction steps](../../experiments/crownless-conversation/README.md)
and the [actual six-turn exchange](../../experiments/crownless-conversation/chat.json).

The model learned nine authored response tasks over 61 account meanings.
The saved model passed 1,247/1,250 main response checks, 1,124/1,250 new-question
wording checks, 244/244 fixed-account history checks, and 1,250/1,250 original
account checks. The report keeps the errors and the scope of each test.

The Python runtime expands the stored 8-bit matrices to float32. Training
started from the v2 core, retained its tokenizer, and selected update 2,500
by validation loss from a 3,000-update local run.

SHA-256: `244a809aba71679efe129495bb981ca8d7fbe0d9012948a5f88731dabcb9c48b`
