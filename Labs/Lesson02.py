##A Cat or Dog classifier deployed on huggingface

from fastai.vision.all import *
import gradio as gr


# If you trained the lesson-2 dogs-vs-cats model, load_learner needs the
# same labelling function that was used in training. Harmless otherwise.
def is_cat(x):
    return x[0].isupper()


# torch.load directly so any missing-module error is shown instead of
# being swallowed by load_learner
learn = torch.load("model.pkl", map_location="cpu", weights_only=False)
learn.dls.cpu()
labels = learn.dls.vocab  # category names come straight from the model


def classify_image(img):
    pred, idx, probs = learn.predict(PILImage.create(img))
    return {str(labels[i]): float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="fastai image classifier",
    description="Upload an image to see what the model predicts.",
)

demo.launch(show_error=True)