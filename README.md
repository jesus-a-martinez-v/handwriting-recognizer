# Handwriting Recognizer

Handwritten digit recognizer implemented with HOG + SVM.

## Installation

I recommend you have `virtualenv` installed. After that, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just install the requirements:

```bash
pip install -r requirements.txt
```

## Try it

At the root of the project, execute the following command to train a model:

```bash
python train.py -d ./data/digits.csv -m ./models/svm.cpickle
```

And then run your model on some image, like this:

```bash
python classify.py -i images/umbc_zipcode.png -m models/svm.cpickle
```

You'll get an output in the terminal similar to this:

```
I think that number is: 2
I think that number is: 1
I think that number is: 2
I think that number is: 5
I think that number is: 0
```

And an annotated photo, like this:
![Annotated](https://github.com/jesus-a-martinez-v/handwriting-recognizer/blob/master/assets/annotated.png)
