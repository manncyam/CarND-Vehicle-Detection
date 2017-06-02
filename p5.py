'''
    train model, load saved model and do processing
    on video to find vehicle in each frame
'''
from helper import *
from IPython.core.debugger import Tracer


processing_mode = {'training':0, 'processing':1}
mode = processing_mode['processing']

def main():   
    image_path = 'C:/Users/MannKhann/CarND/P5/CarND-Vehicle-Detection/training_data/*/*/*.png' 
    if mode == processing_mode['training']:
        train_model(image_path)
    else:
        X_scaler, svc = load_model()
        
# to prevent the file from executed when it import to other files     
if __name__ == "__main__":
    main()
    
