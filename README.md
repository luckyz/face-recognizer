# Face Recognizer

Implements facial recognition using EigenFaces, FisherFaces y LBPH methods. Finally, you can try each method (separately) from ```facial_recognition.py```.

```
pip install -r requirements.txt
```

```face_grabber.py``` takes photos of interested person and it crops around face, saving at ```data``` folder with given person name.
```train_facial_recognition.py``` creates and trains model from persons saved in the past step.
```facial_recognition.py``` executes recognizing using trained model.

# Quickstart
```python
python face_grabber.py
```
> Note: follow app instructions

```python
python train_facial_recognition.py
```

```python
python facial_recognition.py
```
