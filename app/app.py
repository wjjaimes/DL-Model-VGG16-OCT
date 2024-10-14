import os
import numpy as np
import dearpygui.dearpygui as dpg
from tensorflow import keras
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import urllib.request
import cv2
        
def open_file(sender, data):
    # Open file explorer
    with dpg.file_dialog(height=300,directory_selector=False,callback=change_image):
        dpg.add_file_extension(".jpeg")

def change_image(sender, data):
    # Change image window
    dpg.configure_item("cto_image", texture_tag=load_image(data['file_path_name']))

def save_file(sender, data, user_data):
    # Save a .txt file with the user data
    patient_data = dpg.get_values(user_data["Patient_data"])
    vital_signs = dpg.get_values(user_data["Vital_signs"])
    medications = dpg.get_values(user_data["Medications"])
    diagnosis = dpg.get_value(user_data["Diagnosis"])

    if diagnosis:
        with open('results.txt', 'w') as f:
            # Save Patient Data
            f.write(f"Patient data:\n")
            f.write(f"\tName: {patient_data[0].capitalize()}\n")
            f.write(f"\tSex: {patient_data[1]}\n")
            f.write(f"\tDate of Birth: {patient_data[2]}\n")
            f.write(f"\tPhone Number: {patient_data[3]}\n")
            f.write(f"\tEmergency Contact Name: {patient_data[4].capitalize()}\n")
            f.write(f"\tEmergency Contact Phone Number: {patient_data[5]}\n\n")

            # Save Vital Signs
            f.write(f"Vital Signs:\n")
            f.write(f"\tDiastolic blood pressure: {vital_signs[0]} mmHg\n")
            f.write(f"\tSystolic blood pressure: {vital_signs[1]} mmHg\n")
            f.write(f"\tBody height: {vital_signs[2]} cm\n")
            f.write(f"\tBody weight: {vital_signs[3]} kg\n")
            f.write(f"\tHeart rate: {vital_signs[4]} bpm\n")
            f.write(f"\tRespiratory rate: {vital_signs[5]} bpm\n")
            f.write(f"\tBody temperature: {vital_signs[6]} °C\n")
            f.write(f"\tPulse oximetry: {vital_signs[7]}%\n\n")

            # Save Medications
            f.write(f"Medications:\n")
            f.write(f"\tMedications: {medications[0]}\n")
            f.write(f"\tAllergies: {medications[1]}\n\n")

            # Save Diagnosis
            f.write(f"Diagnosis:\n")
            f.write(f"\t{diagnosis}\n")

            # Save
            save_msg = 'Your diagnosis has been saved successfully.'
            show_info('Saved', save_msg, close_show_info)
    else:
        warn_msg = 'Classify an OCT image before saving.'
        show_info('Warning', warn_msg, close_show_info)

def load_image(image_path):
    global img_path 

    img_path = image_path
    width, height, channels, data = dpg.load_image(image_path)

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)

    return texture_id

def show_date(sender, data):
    dpg.configure_item("date", default_value=f"{data['month_day']}/{data['month']+1}/{data['year']+1900}")

def classify(sender, data, user_data):
    model_loaded = user_data[0]
    if model_loaded:
        if img_path == "resources/DME-1.jpeg":
            error_msg = "Load an OCT image before classifying."
            show_info('Warning', error_msg, close_show_info)
        else:
            model = user_data[1]
            to_tf = test_cnn(model,img_path)
            dpg.configure_item("diag_text", default_value=to_tf)
    else:
        error_msg = user_data[1]
        show_info('Error', error_msg, close_show_info)
        dpg.configure_item("diag_text", default_value='Error')


def show_info(title, message, selection_callback):
    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True) as info_id:
            dpg.add_text(message)
            dpg.add_button(tag="ok_id", label="Ok", width=75, user_data=(info_id, True), callback=selection_callback)

    # Centrar popup
    dpg.split_frame()
    width = dpg.get_item_width(info_id)
    height = dpg.get_item_height(info_id)
    dpg.set_item_pos(info_id, [viewport_width//2 - width//2, viewport_height//2 - height//2])

def close_show_info(sender, data, user_data):
    dpg.delete_item(user_data[0])

def get_model():
    try:
        model_name = 'DL-Model-VGG16-OCT.h5'
        _dir = os.getcwd() + '\\' + model_name
        url = 'https://www.dropbox.com/s/6nxdwu2tlqfzl83/' + model_name + '?dl=1'
        # url = 'error'
        urllib.request.urlretrieve(url, _dir)
        model = tf.keras.models.load_model(os.getcwd() + '\\' + model_name)
        os.remove(os.getcwd() + '\\' + model_name)
        return (True, model)

    except Exception as ex:
        error_msg = 'Download of \'' + model_name + '\' failed --> {}.'.format(ex)
        return (False, error_msg)         

def test_cnn(model, file_path, im_size = 496):
    # Classify image
    im_in = cv2.imread(file_path)
    im_resize = cv2.resize(im_in, (im_size, im_size))
    im_out = im_resize.reshape(-1, im_size, im_size, 3)
    out = model.predict(im_out, verbose = 0)
    
    if out.argmax() == 0:
        diagnosis = 'Choroidal Neovascularization (score = ' + str(round(100.0*out[0][0], 2)) + ' %).'
    elif out.argmax() == 1:
        diagnosis = 'Diabetic Macular Edema (score = ' + str(round(94.9*out[0][1], 2)) + ' %).'
    elif out.argmax() == 2:
        diagnosis = 'Drusen (score = ' + str(round(100.0*out[0][2], 2)) + ' %).'
    elif out.argmax() == 3:
        diagnosis = 'Healthy (score = ' + str(round(100.0*out[0][3], 2)) + ' %).'
    
    return diagnosis

def grad_cam(model, img_path, layer_name="block5_conv3"):
    # Cargar y preprocesar la imagen
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    
    # Calcular las activaciones y el gradiente
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Superponer el heatmap en la imagen original
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Guardar la imagen superpuesta
    output_path = 'grad_cam_output.jpg'
    cv2.imwrite(output_path, superimposed_img)
    return output_path

def generate_grad_cam(sender, data, user_data):
    model = user_data[0]
    grad_cam_image = grad_cam(model, img_path)
    dpg.configure_item("grad_cam_image", texture_tag=load_image(grad_cam_image))

# Create context
model = get_model()

dpg.create_context()
dpg.create_viewport(title="Retinal Diseases Detection", width=915, height=410)
dpg.set_viewport_large_icon("resources/icon.ico")
dpg.setup_dearpygui()

dpg.show_viewport()

# Fonts
with dpg.font_registry():
    # Definición de fuentes
    title_font = dpg.add_font("fonts/Roboto-Medium.ttf", 20)
    text_font = dpg.add_font("fonts/Roboto-Regular.ttf", 18)

# Information window
with dpg.window(label="Information",tag="Primary Window",width=500,height=270,no_resize=True,no_collapse=True, no_close=True, no_move=True, no_title_bar=True):
    with dpg.tab_bar():
        with dpg.tab(label='Patient Data'):
            with dpg.group(width=-1):
                # Patient name
                p_name = dpg.add_input_text(label="Name", hint="Name")

                # Patient Sex
                p_sex = dpg.add_combo(("Male", "Female", "Other"), label="Sex", default_value="Sex")

                # Patient Date of birth
                p_date = dpg.add_input_text(tag='date', label="Date of Birth", hint="Date of birth", readonly=True) 
                with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left, modal=False, tag="modal_id", no_move=True):
                    # dpg.add_button(label="Close", callback=lambda: dpg.configure_item("modal_id", show=False))
                    dpg.add_date_picker(callback=show_date)

                # Patient Phone number
                p_num = dpg.add_input_text(label="Phone Number", decimal=True, hint="Phone number")

                # Emergency Contact Name
                e_name = dpg.add_input_text(label="Emergency Contact Name", hint="Emergency contact name")

                # Emergency Contact Phone Number
                e_num = dpg.add_input_text(label="Emergency Contact Phone Number", decimal=True, hint="Emergency contact phone number")


        with dpg.tab(label="Vital Signs"):
            with dpg.group(width=230):
                # Patient Diastolic Blood Pressure
                p_dbp = dpg.add_input_int(label="Diastolic blood pressure (mmHg)", min_value=0, min_clamped=True)

                # Systolic Diastolic Blood Pressure
                p_sbp = dpg.add_input_int(label="Systolic blood pressure (mmHg)", min_value=0, min_clamped=True)

                # Body height
                p_height = dpg.add_input_int(label="Body height (cm)", min_value=0, min_clamped=True)

                # Body Weight
                p_weight = dpg.add_input_float(label="Body weight (kg)", format="%.1f", min_value=0, min_clamped=True)

                # Heart rate
                p_hr = dpg.add_input_int(label="Heart rate (bpm)", min_value=0, min_clamped=True)

                # Respiratory rate
                p_rr = dpg.add_input_int(label="Respiratory rate (bpm)", min_value=0, min_clamped=True)

                # Body Temperature
                p_bt = dpg.add_input_float(label="Body temperature (°C)", format="%.1f", min_value=0, min_clamped=True)

                # Pulse Oximetry
                p_po = dpg.add_input_float(label="Pulse oximetry (%)", format="%.1f", min_value=0, min_clamped=True)

        with dpg.tab(label="Medications"):
            # Medications
            dpg.add_text("Medications")
            p_med = dpg.add_input_text(label="Medications", width=-1, multiline=True, height=80, tab_input=True)

            # Allergies
            dpg.add_text("Allergies")
            p_all = dpg.add_input_text(label="Allergies", width=-1, multiline=True, height=80, tab_input=True)

# Diagnosis window
with dpg.window(label="Diagnosis",width=500, height=100, pos=(0,270),no_resize=True,no_collapse=True, no_close=True, no_move=True):
    # Diagnosis
    hint_text = 'Your diagnosis will show here after selecting the classify button.'
    p_diag = dpg.add_input_text(label="input text", tag='diag_text', multiline=False, width=-1, height=50, tab_input=True, hint=hint_text, readonly=True)

# OCT image window
with dpg.window(label="OCT Image",width=400, height=370, pos=(500,0),no_resize=True,no_collapse=True, no_close=True, no_move=False):
    dpg.add_image(load_image("resources/def_img.jpeg"), tag='cto_image', width=400, height=300)

    with dpg.group(horizontal=True):
        # Load Image
        dpg.add_button(label="Load Image", width=95, callback=open_file)
        # Classify Image
        dpg.add_button(label="Classify", width=95, user_data=model, callback=classify)  
        dpg.add_button(label="GradCAM", width=95, user_data=model, callback=generate_grad_cam)
        # Save
        dpg.add_button(label="Save", width=94, callback=save_file, 
                        user_data={
                                "Patient_data": [p_name, p_sex, p_date, p_num, e_name, e_num],
                                "Vital_signs": [p_dbp, p_sbp, p_height, p_weight, p_hr, p_rr, p_bt, p_po],
                                "Medications": [p_med, p_all],
                                "Diagnosis": p_diag})
        


# Viewport configuration
dpg.bind_font(text_font)
dpg.start_dearpygui()
dpg.destroy_context()
