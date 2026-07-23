# Articles

Articles about Myth Engine's design, architecture, or implementation details.

## All Articles

### [Building an SSA-Based Declarative Render Graph](/articles/render-graph-design)

The full design journey of Myth's render graph compiler: from a linear hardcoded prototype, through a failed "blackboard" pattern, to a strict declarative RenderGraph built on **SSA (Static Single Assignment)**. Covers the compiler lifecycle, automatic memory aliasing, dead-pass elimination (DPE), plus ~1.6µs per-frame compilation benchmarks and auto-generated topology graphs.

---

> Looking for the concise, user-facing architecture overview? Read the [Render Graph](/architecture/render-graph) chapter.
