Predictive KV Cache Eviction for Block-Sparse Attention  
Justin Kim (yutarkk), Cody Lejang (clejang), Clarie Liu (claireli)

**Introduction**  
Large Language Models (LLMs) are widely and increasingly used for tasks that require long context, such as analyzing codebases and retrieving documents. However, the Key-Value (KV) cache is still a significant memory bottleneck. Block-Sparse Attention (BSA) reduces computation overhead by attending to a subset of tokens; the system still must manage which blocks of these tokens can be stored in GPU or CPU memory.  
**Problem**:   
Can we use the attention patterns of early Transformer layers to predict and prefetch the KV blocks for deeper layers? Existing systems like BSAKV rely on an exponential moving average of momentum to determine which blocks to keep in the GPU. However, this approach only knows a block is important after it has been used. In a sequential decoder, this creates a bottleneck where, at each layer, the GPU may have to wait for the CPU to send a block that the model just realized it needs. We aim to solve the latency and accuracy gap in KV cache management by turning this reactive caching policy into a predictive one.  
**Status quo**  
PagedAttention (Kwon et al., 2023\) and vLLM established paged KV cache management for dense attention, but are unaware of BSA’s sparse access patterns. BSAKV introduced a BSA-aligned layout and momentum-decay eviction, outperforming recency-based policies significantly in simulated evaluations, but was evaluated only via trace-driven simulation, with no end-to-end latency measurement. MoBa (Lu et al., 2025\) and NSA (Yuan et al., 2025\) formalized BSA as a training-integrated mechanism, but neither addresses KV cache eviction. Our work improves on BSAKV by introducing cross-layer awareness and validating in a real serving environment.  
Instead of waiting for a block to slowly accumulate momentum across many layers, we will identify early layers in the model that signal which semantic regions the deeper layers will eventually focus on, enabling earlier prefetching.  
**High-level implementation plans**:   
We will integrate BSAKV’s layout into SGLang by implementing a custom KV Cache manager supporting (layer, head, block)-granularity. We evaluate on LongBench and Needle-in-a-Haystack benchmarks using A MoBA-compatible model.  
Our core contribution is a hybrid scoring function that replaces BSAKV’s pure momentum decay with a combination of signals:

- Semantic relevance: EMA of past attention scores, preserving BSAKV’s original intuition  
- Layer-lookahead signal: if layer L’s gating selects a block for L+1, boost that block’s score preemptively  
- Shallow-layer attention: if early layers attend strongly to a block, treat it as a hint for deeper layers.  
- Transfer cost: blocks are expensive to fetch from the CPU, and receive a higher incentive to stay on the GPU.