# Face Recognizer

Facial recognition implementing EigenFaces, FisherFaces y LBPH methods. Finally, you can try each method (separately) from ```facial_recognition.py```.

```face_grabber.py``` takes photos of interested person and it crops around face, saving at ```data``` folder with given person name.
```train_facial_recognition.py``` creates and trains model from persons saved in the past step.
```facial_recognition.py``` executes recognizing using trained model.

## Quickstart

Install dependencies:

```
pip install -r requirements.txt
```

To record and take a face using video streaming:

```python
python face_grabber.py
```
> Note: follow app instructions

Then, train a model:

```python
python train_facial_recognition.py
```

Finally, start face recognizer:

```python
python facial_recognition.py
```
