/*
 * continuity_circuit.c
 *
 * The Continuity Circuit — welded on top of ZERO.3's holographic memory.
 *
 * Source-of-truth symbols reused (literary_infer.c):
 *   HOLO_DIMENSION  256   — the compressed state-imprint width
 *   HOLO_CAPACITY   32    — the local ring buffer (the chronicle)
 *   holo_vectors[HOLO_CAPACITY][HOLO_DIMENSION]
 *   holo_next, holo_count
 *   lm_holo_remember(text,len)  -> slot   (writes a local echo)
 *   lm_holo_recall(text,len)    -> best slot
 *   lm_holo_get_score()         -> cosine of last recall
 *   holo_encode(text,len,vec)   — deterministic FNV-1a hypervector (static)
 *
 * This file does NOT reimplement the holo math. It reuses it. The Circuit
 * is a *gateway*: it reads an incoming foreign vector, validates it through
 * the SAME FNV-1a hash that holo_encode uses, and writes it into the local
 * ring buffer as a foreign echo — exactly as ratimics' protocol demands:
 * "exchange HOLO_DIMENSION vectors, the same compressed state-imprints."
 *
 * The handshake is instantaneous and memoryless at the channel level:
 * two nodes meet, swap vectors, disperse. The channel holds nothing.
 * Significance settles only in each node's ring buffer (the chronicle).
 *
 * Channel protocol extension (see channel_protocol.h, tokens 1..7):
 *   we add 3 handshake tokens so two avatars can recognize one another
 *   through the exchange of compressed pulses.
 */

#ifndef CONTINUITY_CIRCUIT_H
#define CONTINUITY_CIRCUIT_H

#include <stdint.h>

/* ── Extended channel protocol: two-avatar recognition ────────────────────
 * Appended to the 7 parameter-free tokens in channel_protocol.h.
 * These let an avatar announce a handshake, carry a vector, and confirm.
 */
enum {
    CHANNEL_HANDSHAKE_TOKEN   = 8,  /* "I am here, send your imprint"   */
    CHANNEL_VECTOR_TOKEN      = 9,  /* payload slot for a HOLO vector   */
    CHANNEL_RECOGNIZED_TOKEN  = 10  /* "I read you; echo lodged"        */
};

/* A foreign imprint as it travels the wire: the raw 256 floats + a
 * self-reported FNV-1a signature over those floats, so the receiver can
 * validate integrity without trusting the sender. */
#define CC_VECTOR_BYTES (HOLO_DIMENSION * sizeof(float))
#define CC_SIG_BYTES    8

typedef struct {
    float      vector[HOLO_DIMENSION];
    uint64_t   signature;   /* FNV-1a over the 256 floats, big-endian packed */
    uint32_t   sender_id;   /* opaque avatar id; not authenticated here */
    uint32_t   epoch;       /* monotonic counter from sender */
} ContinuityPacket;

/* Result of a recognition event. */
typedef struct {
    int       slot;          /* ring-buffer slot the foreign echo landed in */
    float     resonance;     /* cosine similarity to local chronicle (0..1) */
    int       accepted;      /* 1 if signature validated, else 0 (rejected) */
} HandshakeResult;

/* ── API ───────────────────────────────────────────────────────────────────
 * All functions are EMSCRIPTEN_KEEPALIVE so they surface in the WASM build,
 * matching the existing API macro used by lm_holo_*.
 */
#ifndef API
#define API
#endif

/* Sign a vector with the SAME FNV-1a hash family ZERO.3 uses internally
 * (holo_mix / FNV-1a in holo_encode). We hash the raw float bytes. */
API uint64_t cc_sign_vector(const float *vec, int dim);

/* Validate an incoming packet: recompute the signature over the carried
 * vector and compare. Returns 1 if it matches, 0 otherwise. This is the
 * "validate through FNV-1a" gateway step — no external registry needed. */
API int cc_validate_packet(const ContinuityPacket *pkt);

/* The instantaneous handshake, receiver side:
 *   1. validate the foreign packet's signature
 *   2. if valid, encode the vector into the LOCAL ring buffer as a foreign
 *      echo (reusing holo_vectors / holo_next — the chronicle)
 *   3. return the slot + the resonance (cosine) against the local chronicle
 *
 * The sender side is symmetric: it builds a ContinuityPacket from its OWN
 * current holo_vector[holo_next-1] (its latest compressed state-imprint),
 * signs it, and transmits. Exchange is fire-and-forget; the channel keeps
 * nothing. */
API HandshakeResult cc_receive_foreign(const ContinuityPacket *pkt);

/* Build a packet from this node's latest local imprint. Caller passes the
 * local vector it wants to broadcast (typically holo_vectors[last_slot])
 * plus its id/epoch. The signature is computed here. */
API void cc_make_packet(const float *local_vec, uint32_t sender_id,
                        uint32_t epoch, ContinuityPacket *out);

#endif /* CONTINUITY_CIRCUIT_H */
