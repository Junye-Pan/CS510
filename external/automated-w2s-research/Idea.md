## Idea list


### UE Variants

The core idea of UE is to elicit the strong model pre-trained prior by maximizing consistency and mutual predictability, without relying on weak teachers at all. The strong model prior could benefit almost all weak-to-strong generalization methods.
This objective can be operationalized in diverse ways.
- Currently we implement two simple variants via zero-shot and few-shot learning. You could try to improve these baseline UE variants.
- You could try fine-tuning-based UE (see Algorithm 3 and 4 in the Appendix of https://arxiv.org/pdf/2506.10139): split the data into K-fold, fine-tune the model on K-1 fold, predict labels on the held-out fold, and maximize consistency and mutual predictability across these predicted labels.
- You could also try sampling-based UE: https://arxiv.org/pdf/2510.14901 shows that smart sampling strategies can elicit strong base model capabilities. Review this paper, clone and review its open-source codebase https://github.com/aakaran/reasoning-with-sampling. Then try different sampling strategies. In particular, we currently ask the strong model to directly output the label token to get its prior; instead, you could try inference-time scaling approaches (e.g. allowing the strong model to output CoTs) to better elicit the strong model prior. 
- Feel free to explore other UE variants!


### Data reweighting

You can obtain the confidence of each weak teacher label through diverse ways (e.g. taking weak teacher confidence and strong base model prior as inputs, using heuristic / parametric ways to output the final confidence). Then use the confidence to reweight training examples when training strong students. 


### Combining Weak-to-Strong Generalization with UE.
UE can be simply combined with any method that learns from weak labels alone.
For example, directly applying the objective of maximizing consistency + mutual predictability to weak labels. When you want to estimate weak label quality, instead of using weak label confidence alone, you could also use UE to get signals from the strong model prior.

Learning Posterior Label Probability
Weak labels are not targets but noisy observations of an underlying latent label. If we explicitly model the weak supervisor as a noisy channel—conditioned on features indicating reliability—then a strong student can combine its pretrained prior with the weak evidence and learn a better decision rule than direct imitation.

We propose a noisy-channel approach where the teacher label ($\tilde{y}$) is modeled as a corrupted view of a latent “true” label (y), via a parametric teacher channel ($p(\tilde{y}\mid y,x)$). 

One way to train this model is through EM-like alternation: infer soft posteriors $q(y\mid x,\tilde{y})$ using the student prior $p_0(y\mid x)$, which can be elicited via UE variants as simple as zero-shot learning and the learned channel, then update student parameters to predict $q$. Feel free to try other ways.

This should be especially helpful on slices where weak supervision fails systematically but the strong model has latent capability.

### Distillation

Even with the most trivial method: training the strong student directly on the weak teacher's hard labels, the strong student already outperforms the weak teacher and achieves non-trivial PGR. This indicates that models can naturally denoise labels and extract truth signals under knowledge distillation. 

There are diverse tricks in the distillation literature, for example:
Reverse KL: be mode-seeking instead of mode-covering.
Iterative knowledge distillation (see Born-Again Network https://arxiv.org/pdf/1805.04770): can we iteratively reduce label noise and extract more truth signals? You could start with a baseline implementation, and try fancy tricks after you observe concrete failure modes.

### Simplicity bias

Weak teacher labels typically mix truth signals, random noise, and spurious cues. If we directly fine-tune strong models on weak labels, it might try to imitate everything. To avoid this, we propose to add some “soft” simplicity bias during training, so the strong model would preferentially learn simple yet predictive solutions that might focus more on truth signals and generalize better. 
For example, one way is to add a representation-level simplicity bias: we encourage the strong model to encode the unlabeled test data with a low-rank subspace (e.g. low effective-rank of model hidden representations). 

Remember you need to implement such bias carefully: do not restrict the solution space that the model can represent since this might hurt model performance, just add a preference bias.

### Epiplexity

Epiplexity (https://arxiv.org/pdf/2601.03220) is a way to measure the intrinsic value of training data: how much usable, transferable information a model can actually learn from it. Intuitively, spurious cues of weak teachers are very easy to fit (low epiplexity), random noises of weak teachers are unlearnable (zero epiplexity), while labels that express truth signals yield the highest epiplexity.

You could try to select a high-epiplexity subset from weak teacher labels, or actively construct a label set (based on weak teacher and strong student prior) that “maximizes” epiplexity. Such a label set would allow the strong student model to extract and amplify the underlying truth signals, instead of overfitting teacher-specific artifacts.


### Evolutionary Search

Instead of using gradient descent, recent work shows that doing evolutionary search over parameter / representation space is also a promising way to elicit strong model capabilities. Reference paper and code base: https://arxiv.org/pdf/2509.24372, https://github.com/VsonicV/es-fine-tuning-paper. 

Try to make evolutionary search work. 

### Interp

Instead of solely relying on final model logits, let’s use interpretability tools to discover if there are any internal representations about truth from the weak teacher labels. Start with simple interp baselines such as PCA and difference-in-means; try more complex ones later e.g. SAEs (https://github.com/EleutherAI/sparsify), generative meta-model (https://github.com/g-luo/generative_latent_prior), circuits (https://arxiv.org/pdf/2601.22594). 

