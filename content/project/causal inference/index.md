---
slides: example
url_pdf: ""
summary: ""
url_video: ""
date: 2020-10-06T00:00:00.000Z
external_link: ""
url_slides: ""
title: Causal Inference
tags: []
links: []
image:
  caption: ""
  focal_point: Smart
url_code: ""
---


This is a blog for my project on the applications of causal inference to machine learning. 

------------------------------

<strong>What is causal inference and why do I care?</strong> 

What do we really mean when we say "the economy does better when my favorite political party is in power" or what do we really mean when we say ["unvaccinated adults are 3-5 times more likely to get omicron infection"](https://www.aha.org/news/headline/2022-01-21-cdc-unvaccinated-adults-3-5-times-more-likely-get-omicron-infection#:~:text=According%20to%20data%20from%2025,who%20had%20received%20a%20booster). A very natural property of interest in any dataset is the pattern of [correlation](https://en.wikipedia.org/wiki/Correlation) - when two events are statistically related. Correlation is an important part of statistics, but what's more important is how we tend to interpret it. Correlations tell us that two events tend to share some pattern and not that one event is the reason for the other occuring. We really want to say that "the economy is better *because* of my favorite political party, and not just they happen to occur together. Establishing causal relationships between phenomenon is the study of [causal inference](https://en.wikipedia.org/wiki/Causal_inference#:~:text=Causal%20inference%20is%20the%20process,component%20of%20a%20larger%20system.). You might have heard the phrase 'correlation is not the same as causation'. This is true. While it's true that both the number of tech companies and the number of teddy bears sold has gone up over the last few decades, one does not necessarily cause the other (perhaps both are caused by, among lots of other things, an increase in population sizes leading to more children buying teddy bears and a bigger scientific community built over the years). But this is an example of events that are correlated - they both values have gone up together - but probably not causally related. 

I'm interested in causal inference, at the highest level, because it is, in some sense, the most natural question to ask when studying complex datasets. Questions about causation are often the most interesting and, so, inherently the most enlightening in any survey. We know when the economy rises and falls but it's establishing its cause thats the most interesting. Mathematically, this presents much richer questions: how does one quantitatively establish causation between variables? how is causation structurally different from phenomenon to phenomenon? and how do we quantify that difference? These questions have a lot of implication on not only our understanding of economic and natural phenomenon but also about the human mind and our understanding of how we learn things and form such associations and establish legitimate and unfound causal relationships. I'm interested in these sorts of questions and am looking to explore these in greater depth because I'm fascinated by the question of causation, the mathematical bases on which it can be established, and the understanding of human behavior and learning patterns that this brings. 


-------------------------------

<strong>What question am I interested in?</strong>

I'm interested in the applications of causal inference to machine learning (ML). The main aim for any ML problem is to predict a label <strong> y </strong> from a feature vector <strong> x </strong> by training an algorithm on some known features. Different approaches require different training data: <em><strong> supervised learning </strong></em> refers to the method of training algorithms using feature vectors with known labels, while <em><strong> semisupervised learning </strong></em> (SSL) refers to the method of training algorithms with some data with known labels and some with unknown labels. The underlying assumption in most ML problems is that we are trying to find <em> associations or correlations</em> between the labels and features to make predictions; we normally don't care about their causal structure. It doesn't seem to matter if <strong> x </strong> causes <strong> y </strong> or vice versa since algorithms are expected to perform the same. However, this has shown to not necessarily be true. Scholköpf et al. showed in their [2012 ICML paper](https://icml.cc/2012/papers/625.pdf) that there is in fact a difference in performance in these two cases. SSL algorithms performed better on <em>anticausal problems</em> (i.e. where <strong> x </strong> causes <strong> y </strong>) than in <em>causal problems</em> (vice versa). Figure 1 below illustrates this.  

{{< figure src="ICML_data.png" caption="The benefit of SSL depends on the causal structure. Each column of points corresponds to a benchmark data set from the UCI repository and shows the performance of six different base classifiers augmented with self-training, a generic method for SSL. Performance is measured by percentage decrease of error relative to the base classifier, that is, (error(base)−error(self-train))/error(base). Self-training overall does not help for the causal data sets, but it does help for some of the anticausal confounded data sets [from [Scholköpf et al. 2012](https://icml.cc/2012/papers/625.pdf)]." numbered="true" >}}

Here we see that the performance of SSL on anticausal problems and causal is considerably different to supervised learning. I'm interested in looking at why this is the case and what kinds of problems do better with SSL vs supervised learning.

A toy example where we can see this is in the case of two variables

<math>$$X = N_1$$</math> 
<math>$$Y = N_1 + X$$</math>

Where <math> $N_1, N_2 \overset{\mathrm{iid}}{\sim} Be(\frac{1}{2})$</math>. Here there is a clear causal dependency  <math>$X \rightarrow Y$</math>.

We can factor the joint probability distribution as

<math>$$P(X,Y) = P(X) P(Y|X)  = P(Y) P(X|Y)$$</math>

This factorization is not symmetric since we know that with the current variables, we get the probabilities 

<math>$$P(X = x) = 1/2 \text{ for } x=0,1$$</math>
<math>$$P(Y=y) = \begin{cases} \frac{1}{4} & \text{if }y=0,2 \newline \frac{1}{2} & \text{if }y=1\end{cases}$$</math>
<math>$$P(X=x | Y = 0) = \begin{cases} 1 & \text{if }x=0 \newline 0 & \text{if }x=1\end{cases}$$</math>
<math>$$P(Y=y | X = 0) = \begin{cases} \frac{1}{2} & \text{if }y=0 \newline \frac{1}{2} & \text{if }y=1 \newline 0 & \text{if }y=2\end{cases}$$</math>

In ML, we're usually trying to understand the distribution <math> $ P(Y|X)$ </math>, i.e. given the training data, what is the probability of having any given label. But notice that when we change the distribution of X to, say, <math>$Be(\frac{2}{3})$</math>, then we get the probabilities 

<math>$$P(X=x) = \begin{cases} \frac{1}{3} & \text{if }x = 0 \newline \frac{2}{3} & \text{if }x=1\end{cases}$$</math>
<math>$$P(Y=y) = \begin{cases} \frac{1}{4} & \text{if }y=0,2 \newline \frac{1}{2} & \text{if }y=1\end{cases}$$</math>
<math>$$P(X=x | Y = 0) = \begin{cases} 1 & \text{if }x=0 \newline 0 & \text{if }x=1\end{cases}$$</math>
<math>$$P(Y=y | X = 0) = \begin{cases} \frac{1}{3} & \text{if }y=0 \newline \frac{2}{3} & \text{if }y=1 \newline 0 & \text{if }y=2\end{cases}$$</math>

Notice that the distribution for <math> $ P(X|Y)$ </math> does not change but the distribution for <math> $ P(Y|X)$ </math> does. This is a two-variable example to show the assymetry in causal vs anticausal problems and why certain machine learning algorithms might learn  <math> $ P(Y|X)$</math> differently in these cases. The goal for this project is to study this phenomenon. Specifically, I'm interested in the following questions:

1. Can we prove that SSL performs better in anticausal learning problems than other techniques, even for specific cases?

2. Is there an example of a dataset that is traditionally studied with ML techniques agnostic of causal structure that might be better suited to SSL techniques due to an underlying causal structure? 

--------------------------------

<strong>Identifiability</strong>

To understand this phenomenon better, first, let's focus on the data and go back to our original motivation: discovering causal structure and finding the causal relationships between different variables. Given features <math>X</math> and labels <math>Y</math>, can we distinguish between the causal directions <math>$X \rightarrow Y$</math> and <math>$Y \rightarrow X$</math>? That is to say, can we ascribe a deterministic <em>mechanism</em> <math> $X \xrightarrow{\varphi} Y$ as apposed to $Y \xrightarrow{\psi} X$ </math>. 

A basic assumption that we will make is that each variable is affected by (independent) noise variables <math>$N_Y$ (and $N_X$ for the reverse causal structure)</math>. So, the model we will work with is the graph in figure 2. 

{{<figure src="causal_graph_1.png" caption="Causal Graph from X to Y" numbered="true">}}


This is a basic <em>structural causal model</em> (SCM) - a model which includes a graph with one vertex per variable, jointly independent noise variables, and a deterministic function for each vertex that depends on the vertex's parents and noise. This tells us we have a deterministic mechanism <math>$\varphi$</math> that gives us <math>$Y$ as $Y = \varphi(X, N_Y)$</math>. The question of identifying graphs in <em>causal learning</em> then becomes a question of whether we can ascribe another mechanism from <math>$Y$ to $X$</math> as shown in figure 3. 

{{<figure src="causal_graph_2.png" caption="Causal Graph from Y to X" numbered="true">}}


It turns out that without any restrictions on the mechanism, the problem of <em>identifying</em> the graph is not feasible. Given variables <math>$X,Y$</math> along with their probability distribution <math>$P(X,Y)$</math>, you can always fit a <em>mechanism</em> <math>$\varphi$</math> for either $X \rightarrow Y$ or $Y \rightarrow X$ (the noise variables being understood to be embedded in the graph). This is in the following proposition which states:

<strong>Proposition 1.</strong> Given two random variables <math>$X$ and $Y$</math>, there is a (measurable) function <math>$f$</math> and random variable <math>$N_Y$</math> such that <math>$Y = f(X, N_Y)$</math>

{{<spoiler text="Click to see proof">}}
This isn't the main proof of this section, so I'll refer you to [this paper](http://ei.is.mpg.de/uploads_file/attachment/attachment/2/ACM_Zhang14.pdf) by Zhang et al. for a nice proof. See Lemma 1. 
{{</spoiler>}}

Without any conditions, the indentification problem is not feasible. This should intuitively feel good - without any assumptions, we have no control over how sensitive the mechanism is to the noise and hence the complexity of <math>$P(Y|X)$</math>.

The question thus becomes: Under what conditions can we  We will make the following assumptions:

1. The mechanism is an <em><strong>Additive Noise Model</strong></em> (ANM), i.e. <math>$\varphi (X, N_Y) = \phi(X) + N_Y$</math> for some function <mathb>$\phi$, where $N_Y$ and X </math> are independent. 

2. The usual assumptions of SCMs also hold, such as the independence of the mechanism and input (<math>$P(Y | X)$ and $X$</math>), Markov compatibility (<math>$P(Y)$ just depends on it's parent $X$ - which is trivial in our case</math>), and causal sufficiency. 

The first assumption is our main one. Hoyer et al. showed in their [2009 NeurIPS paper](https://papers.nips.cc/paper/2008/file/f7664060cc52bc6f3d620bcedc94a4b6-Paper.pdf) that under the ANM assumption, we can recover the causal mechanism in most cases. Before we state the main theorem, it is worth noting that since <math>$X$ and $N_Y$</math> are independent, th joint distribution <math>$P(X,Y)$</math> can be rewritten as

<math>$$ P(X = x, Y = y) = P(X = x, N_y = y - \phi(x))$$ $$= P_{X}(X = x) \cdot P_{N_Y}(N_Y = y - \phi(x)) $$</math>

This equivalent restatement will make the theorem easier to state:

<em><strong>Theorem 1.</strong> Given <math>$X, N_Y$, and $Y$</math> that satisfy an ANM with a function <math>$\phi$</math>, if there is a backward mechanism of the same form, then $\phi, P_X, P_{N_Y}$ must satisfy the following differential equation:

<math>$$ \xi'''  = \xi'' \left( -\frac{\nu''' \phi'}{\nu''} + \frac{\phi''}{\phi'} \right) - 2 \nu '' \phi'' \phi' + \nu' \phi''' + \frac{\nu' \nu''' \phi'' \phi'}{\nu''} - \frac{\nu' (\phi'')^2}{\phi'}$$</math>

where <math>$\nu := \operatorname{log} P_{N_Y}$ and  $\xi := \operatorname{log}P_X$</math>, and we also have that <math>$\nu''(y-\phi(x))\phi'(x) \neq 0$ </math>.

Also, we have that if these conditions hold, then if there is a <math>$y$ for which $\nu''(y-\phi(x))\phi'(x) \neq 0$</math> is true for every <math>$x$</math> aside from a countable set, then the set of all <math>$P_X$</math> which admit a backward model is 3-dimensional (i.e. can be contained in a 3-dimensional affine space).</em>

Of course, here we are assuming that all the relevant functions are thrice differentiable. 

{{<spoiler text="Click here if you're interested in the proof">}}
  You can find my write up and notes on the proof {{% staticref "files/Proof_of_Theorem.pdf" %}} here {{% /staticref %}}
{{</spoiler>}}

The differential equation seems quite arbitrary, but let's understand at a high level what this tells us. The differential equation is of order 3 (since the highest derivative taken is a third derivative). We can also see that it is linear in <math>$\xi$</math>; that is, it can be written as 

<math>$$a_0 + a_2 \xi'' + a_3 \xi ''' = b(x,y)$$</math>

and the solution space must be 3-dimensional (see proof for details). Given that the dimension of the space of <em>every possible</em> solution <math>$\xi$</math> is infinite dimensional, this tells us that almost every ANM <math>$\varphi$</math> does not admit a backwards model since most <math>$\phi, P_X, P_{N_Y}$</math> don't satisfy the above differential equation. 

On a side note, a fun corollary of this theorem is that if <math>$\xi''' = \nu''' = 0$</math> everywhere, then the only <math>$\phi$ for which a backward model exists is for linear $\phi$</math>. This tells us that, for example, in the case where <math>$P_X$ and $P_{N_Y}$</math> are Gaussian, only linear <math>$\phi$</math> admit backward models. 

{{<spoiler text="Click here if you're interested in the proof">}}
This proof follows pretty much exactly as in the Hoyer paper, but if you'd like to see it, you can find my write up and notes {{% staticref "files/Proof_of_Corollary.pdf" %}} here {{% /staticref %}} 
{{</spoiler>}}

What this tells us about the identification problem is that if given variables <math>$X,Y$</math>, if we can fit an ANM <math>$\varphi : X \rightarrow Y$</math>, then with probability 1, we can assume that the causal relationship is <math>$X \rightarrow Y$</math>. In the next section (to come), we'll include some experiments to show how we can use this theoretical results in some practical cases with both with real data and synthetic examples. Then, we'll move to studying the SSL problem under these conditions. Stay tuned!

For some further reading for anyone interested, Zhang and Hyvärinen showed in their [2010 AUI paper](https://arxiv.org/pdf/1205.2599.pdf) similar results for postlinear ANM models, which apply a further nonlinear transformation to the ANM we used. The proof techniques used there are quite similar to the proof technique used here!





