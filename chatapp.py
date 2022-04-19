from keras.models import load_model
import json
import pickle
from chatbotgui import ChatbotGUI

def main():
    model = load_model('mychatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))

    gui = ChatbotGUI(model, words, classes, intents)
    gui.loop_gui()
    
if __name__ == "__main__":
    main()