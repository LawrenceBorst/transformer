# Implementation of a Transformer

As practice with Pytorch, among other things, I decided to implement the transformer model according to the seminal paper<sup>[1]</sup> by Ashish Et Al. introducing the model. 

I highly recommend reading the chapter on transformers in Deep Learning<sup>[2]</sup> for an introduction read alongside this paper, offering a motivation for the self-attention equation. The more up-to-date book by Prince<sup>[3]</sup> offers an intuitive, more visual—albeit less detailed—exposition of this piece of math.

# Run the Code
First install packages with
```
poetry install
```

With VSCode, you can simply run the debug scripts provided in the launch.json file, otherwise navigate to your project folder and type

```
poetry shell    # Enter virtual environment

python -m scripts.main
```

# Citations

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, & Illia Polosukhin. (2023). Attention Is All You Need.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

3. Simon J.D. Prince (2023). Understanding Deep Learning. The MIT Press.
