# Bring-up order and acceptance gates

1. **Pin the contract.** Base the fork branch on current upstream, not a stale
   engine release. Compare actual weight names with reference forward calls.
   Never register the 37B model as a stand-in. Reference code and metadata
   suffice for this phase.
2. **Build a debuggable native step.** One sequence, local weights, finite
   context, greedy output. One implementation of each layer is used during both
   prefill and decode. Keep the opt-in target separate from serving defaults.
3. **Prove the graph.** Generate a small randomly weighted instance of the
   official class. Compare full prompt against cached one-token steps, routes
   and layer outputs before comparing logits and generated IDs. Deliberately
   cross the DSA top-k boundary. Check FP8/scales independently of graph parity.
4. **Test storage and SM70.** File reads, host cache and GPU tile cache must
   have explicit byte budgets. Compile for real `sm_70`, then run on a V100.
   Compare CPU/device outputs on the same synthetic tensors. An SM70 binary is
   not evidence that allocation, pinned transfers or inference ran on a card.
5. **Qualify full size.** Check available storage and checkpoint integrity on
   the target. Run a short prompt first, with a cache budget that leaves room
   for transient buffers. Compare against saved reference outputs for the
   *same checkpoint*. Record cold/warm metrics, and cache and transfer traces.
6. **Optimize the measured bottleneck.** Replay routes per layer; evaluate
   persistence vs frequency on held-out tokens. Only enable prefetch/double
   buffering if misses actually overlap computation and do not evict needed
   weights. Test packed experts and latent KV independently for parity.

For an upstream submission, first agree scope with maintainers as required by
`.github/CONTRIBUTING.md`. A GitHub issue/PR search for `GigaChat` at branch
creation found no existing discussion. Keep hardware results and untested
configurations explicit. See [status](DEVELOPMENT_STATUS.md) for completed gates.
