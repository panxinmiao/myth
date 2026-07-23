# 文章

关于 Myth Engine 的设计、架构或实现细节等内容的文章。

## 文章列表

### [构建基于 SSA 的声明式渲染图](/zh/articles/render-graph-design)

Myth 渲染图编译器的完整设计历程：从线性硬编码原型，到"黑板"模式的失败尝试，最终收敛到一个基于 **SSA（静态单赋值）** 的、严格的声明式 RenderGraph。文中包含编译器生命周期、自动内存别名、死节点剔除（DPE）的原理，以及单帧编译耗时约 1.6µs 的实测基准与自动生成的拓扑图谱。

---

> 想了解面向使用者的精炼版架构说明？请阅读 [Render Graph 渲染图](/zh/architecture/render-graph) 章节。
