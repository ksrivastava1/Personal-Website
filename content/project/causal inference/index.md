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

What is causal inference and why do I care? 

What do we really mean when we say "the economy does better when my favorite political party is in power" or what do we really mean when we say ["unvaccinated adults are 3-5 times more likely to get omicron infection"](https://www.aha.org/news/headline/2022-01-21-cdc-unvaccinated-adults-3-5-times-more-likely-get-omicron-infection#:~:text=According%20to%20data%20from%2025,who%20had%20received%20a%20booster). A very natural property of interest in any dataset is the pattern of [correlation](https://en.wikipedia.org/wiki/Correlation) - when two events are statistically related. Correlation is an important part of statistics, but what's more important is how we tend to interpret it. Correlations tell us that two events tend to share some pattern and not that one event is the reason for the other occuring. We really want to say that "the economy is better *because* of my favorite political party, and not just they happen to occur together. Establishing causal relationships between phenomenon is the study of [causal inference](https://en.wikipedia.org/wiki/Causal_inference#:~:text=Causal%20inference%20is%20the%20process,component%20of%20a%20larger%20system.). You might have heard the phrase 'correlation is not the same as causation'. This is true. While it's true that both the number of tech companies and the number of teddy bears sold has gone up over the last few decades, one does not necessarily cause the other (perhaps both are caused by, among lots of other things, an increase in population sizes leading to more children buying teddy bears and a bigger scientific community built over the years). But this is an example of events that are correlated - they both values have gone up together - but probably not causally related. 

I'm interested in causal inference, at the highest level, because it is, in some sense, the most natural question to ask when studying complex datasets. Questions about causation are often the most interesting and, so, ineherently the most enlightening in any survey. We know when the economy rises and falls but it's establishing its cause thats the most interesting. Mathematically, this presents much richer questions: how does one quantitatively establish causation between variables? how is causation structurally different from phenomenon to phenomenon? and how do we quantify that difference? These questions have a lot of implication on not only our understanding of economic and natural phenomenon but also about the human mind and our understanding of how we learn things and form such associations and establish legitimate and unfound causal relationships. I'm interested in these sorts of questions and am looking to explore these in greater depth because I'm fascinated by the question of causation, the mathematical bases on which it can be established, and the understanding of human behavior and learning patterns that this brings. 


-------------------------------

What question am I interested in?

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

In ML, we're usually trying to understand the distribution P(y|x), i.e. given the training data, what is the probability of having any given label. But notice that when we change the distribution of X to, say, <math>$Be(\frac{2}{3})$</math>, then we get the probabilities 

<math>$$P(X=x) = \begin{cases} \frac{1}{3} & \text{if }x = 0 \newline \frac{2}{3} & \text{if }x=1\end{cases}$$</math>
<math>$$P(Y=y) = \begin{cases} \frac{1}{4} & \text{if }y=0,2 \newline \frac{1}{2} & \text{if }y=1\end{cases}$$</math>
<math>$$P(X=x | Y = 0) = \begin{cases} 1 & \text{if }x=0 \newline 0 & \text{if }x=1\end{cases}$$</math>
<math>$$P(Y=y | X = 0) = \begin{cases} \frac{1}{3} & \text{if }y=0 \newline \frac{2}{3} & \text{if }y=1 \newline 0 & \text{if }y=2\end{cases}$$</math>

Notice that the distribution for <math> $ P(X|Y)$ </math> does not change but the distribution for <math> $ P(Y|X)$ </math> does. This is a two-variable example to show the assymetry in causal vs anticausal problems and why certain machine learning algorithms might learn  <math> $ P(Y|X)$</math> differently in these cases. My immediate next steps are to study this problem in three variables and study the datasets from the work of Scholköpf et al. to look into this phenomenon. 

--------------------------------