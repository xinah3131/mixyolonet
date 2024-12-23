import streamlit as st
from PIL import Image
import torch
import os
import argparse
import yaml
import cv2
import random
import time
import pandas as pd
import plotly.graph_objects as go

import utils as util
from train import inference

def load_args_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_path', default='./weights/best1.pt', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_test', action='store_true')
    parser.add_argument('--data', default='rtts', type=str)
    parser.add_argument('--detection_weight', default=0.1, type=int)
    parser.add_argument('--dehazing_weight', default=0.9, type=int)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args([])

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    return args, params

def home_page():
    st.title("Mix-YOLONet: Dehazing and Object Detection")
    st.write("Welcome to the Mix-YOLONet demo application!")
    
    st.markdown("""
    ### Key Features:
    - State-of-the-art dehazing and object detection
    - Performs well in foggy and hazy conditions
    - Outperforms existing models on multiple datasets

    Use the tabs above to explore different sections of the app.
    """)

    st.image("deployment/model/Ynet with attention on decoder crop.png", caption="Sample output from Mix-YOLONet", use_column_width=True)


def model_architecture_page():
    st.header("Model Architecture")
    st.write("This section showcases the architecture of Mix-YoloNet.")

    architecture_image_path = "deployment/model/Ynet with attention on decoder crop.png"

    if os.path.exists(architecture_image_path):
        image = Image.open(architecture_image_path)
        st.image(image, caption="Overall Architecture of Mix-YoloNet", use_column_width=True)
        
        st.markdown("""
        ### Key Components:
        1. **Encoder**: Extracts features from the input image
        2. **Decoder**: Reconstructs the dehazed image
        3. **Detection Head**: Performs object detection on the dehazed image
        
        The model uses attention mechanisms to improve performance in challenging conditions.
        """)
    else:
        st.error(f"Architecture image not found: {architecture_image_path}")

def results_comparison_page():
    st.header("Performance Comparison")
    
    dataset = st.selectbox("Choose a dataset", ["VOC-FOG", "RTTS", "FOGGY DRIVING"], key="results_comparison_dataset")
    results_data = show_table_result(dataset)
    
    st.markdown("""
    ### Key Observations:
    - Mix-YOLONet consistently outperforms other models across different datasets
    - Significant improvement in mAP scores, especially in challenging conditions
    - Demonstrates the effectiveness of our multi-task approach
    """)

    # Create a more customized bar chart using plotly
    df = pd.DataFrame(results_data)
    df = df.sort_values('mAP')  # Sort the dataframe by mAP in ascending order

    fig = go.Figure(go.Bar(
        x=df['mAP'],
        y=df['Model'],
        orientation='h',
        marker_color=['#1f77b4' if model != 'Mix-YOLONet (Ours)' else '#2ca02c' for model in df['Model']],
        text=df['mAP'].round(2),  # Display mAP values on the bars
        textposition='outside'
    ))

    fig.update_layout(
        title=f"mAP Comparison for {dataset} Dataset",
        xaxis_title="mAP",
        yaxis_title="Model",
        height=500,  # Adjust the height of the chart
        width=800,   # Adjust the width of the chart
        margin=dict(l=200, r=20, t=50, b=50)  # Adjust margins to accommodate long model names
    )

    fig.update_xaxes(range=[0, max(df['mAP']) * 1.1])  # Extend x-axis slightly beyond the maximum mAP value

    st.plotly_chart(fig)

def show_table_result(dataset):
    st.subheader(f"Performance Comparison: {dataset}")
    
    if dataset == "FOGGY DRIVING":
        results_data = {
            'Model': ['Yolo V8', 'Yolo V8 (with Clean Images)','IA-YOLO', 'DS-Net','TogetherNet', 'Mix-YOLONet (Ours)'],
            'Type': ['Baseline','Baseline', 'Image Adaptive', 'Multi-task', 'Multi-task','Multi-task'],
            'mAP': [44.27, 39.12, 18.34, 29.47, 34.93, 45.70]
        }
    elif dataset == "RTTS":
        results_data = {
            'Model': ['Yolo V8', 'Yolo V8 (with Clean Images)', 'IA-YOLO', 'DS-Net','TogetherNet',  'Mix-YOLONet (Ours)'],
            'Type': ['Baseline','Baseline', 'Image Adaptive', 'Multi-task', 'Multi-task', 'Multi-task'],
            'mAP': [66.14, 64.90, 35.66, 32.71,61.55, 67.27]
        }
    else:
        results_data = {
            'Model': ['Yolo V8','Yolo V8 (with Clean Images)', 'IA-YOLO', 'DS-Net','TogetherNet',  'Mix-YOLONet (Ours)'],
            'Type': ['Baseline', 'Baseline','Image Adaptive', 'Multi-task', 'Multi-task', 'Multi-task'],
            'mAP': [79.79, 67.19,64.77, 65.89 ,79.10, 80.59]
        }
    df = pd.DataFrame(results_data)

    st.table(df.style.highlight_max(subset=['mAP'], color='lightgreen'))

    st.caption(f"Table: Comparison of Mix-YOLONet with state-of-the-art detection models on the {dataset} dataset.")

    return results_data
def get_qualitative_images(folder_path, dataset, num_images=3):
    random.seed(int(time.time() * 1000))

    all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    yolo_images = [f for f in all_images if "_yolo" in f]
    paired_images = []
    for yolo_img in yolo_images:
        base_name = os.path.splitext(yolo_img)[0].replace("_yolo", "")
        my_img = base_name + '.jpg'
        yologt_img = base_name + '_yologt.jpg'
        if dataset == "FOGGY DRIVING":
            if my_img in all_images and yologt_img in all_images:
                paired_images.append((
                    os.path.join(folder_path, yolo_img),
                    os.path.join(folder_path, yologt_img),
                    os.path.join(folder_path, my_img)
                ))
        else:
            tgt_img = base_name + '_tgt.jpg'
            if my_img in all_images and tgt_img in all_images and yologt_img in all_images:
                paired_images.append((
                    os.path.join(folder_path, yolo_img),
                    os.path.join(folder_path, yologt_img),
                    os.path.join(folder_path, tgt_img),
                    os.path.join(folder_path, my_img)
                ))

    random.shuffle(paired_images)
    
    return paired_images[:num_images]

def qualitative_evaluation_page():
    st.header("Qualitative Evaluation")
    dataset = st.selectbox("Choose a dataset", ["VOC-FOG", "RTTS", "FOGGY DRIVING"], key="qualitative_evaluation_dataset")
    show_qualitative_evaluation(dataset)

def show_qualitative_evaluation(dataset):
    st.subheader(f"Qualitative Evaluation: {dataset}")

    dataset_folders = {
        "VOC-FOG": "qualitative/voc",
        "RTTS": "qualitative/rtts",
        "FOGGY DRIVING": "qualitative/foggy"
    }

    if dataset in dataset_folders:
        folder_path = dataset_folders[dataset]
        button_key = f"randomize_{dataset}"
        
        if st.button("Randomize Images", key=button_key):
            st.session_state[f'random_images_{dataset}'] = get_qualitative_images(folder_path, dataset)
        
        if f'random_images_{dataset}' not in st.session_state:
            st.session_state[f'random_images_{dataset}'] = get_qualitative_images(folder_path, dataset)

        images = st.session_state[f'random_images_{dataset}']
        
        if dataset == "FOGGY DRIVING":
            for yolo_img, yologt_img, my_img in images:
                cols = st.columns(3)
                with cols[0]:
                    st.image(yolo_img, caption='YOLO Result', width=250)
                with cols[1]:
                    st.image(yologt_img, caption='YOLO GT Result', width=250)
                with cols[2]:
                    st.image(my_img, caption='Mix-YOLONet(Ours) Result', width=250)
        else:
            for yolo_img, yologt_img, tgt_img, my_img in images:
                cols = st.columns(4)
                with cols[0]:
                    st.image(yolo_img, caption='YOLO Result', width=200)
                with cols[1]:
                    st.image(yologt_img, caption='YOLO GT Result', width=200)
                with cols[2]:
                    st.image(tgt_img, caption='TogetherNet Result', width=200)
                with cols[3]:
                    st.image(my_img, caption='Mix-YOLONet(Ours) Result', width=200)

    st.markdown("""
    ### Legend:
    - **YOLO Result**: YOLOv8 model trained on hazy images
    - **YOLO GT Result**: YOLOv8 model trained on clean images
    - **TogetherNet Result**: Result from TogetherNet model (for RTTS and VOC-FOG only)
    - **Mix-YOLONet Result**: Our proposed model
    """)

def inference_page(args, params, device):
    st.header("Inference")
    st.write("See Mix-YoloNet in action by uploading an image or choosing a sample!")

    # Create two columns for image source selection
    col1, col2 = st.columns(2)
    
    with col1:
        image_source = st.radio("Choose image source:", ("Upload an image", "Use a sample image"))

    with col2:
        if image_source == "Upload an image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        else:
            sample_dir = os.path.join("deployment", "sample")
            sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            selected_sample = st.selectbox("Choose a sample image:", sample_images)
            uploaded_file = os.path.join(sample_dir, selected_sample)

    # Create two columns for dehazing and confidence threshold
    col3, col4 = st.columns(2)
    
    with col3:
        dehaze_image = st.checkbox("Apply dehazing", value=True)
    
    with col4:
        confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.01)

    if uploaded_file is not None:
        input_dir = os.path.join("deployment", "input")
        os.makedirs(input_dir, exist_ok=True)
        output_dir = os.path.join("deployment", "output")
        os.makedirs(output_dir, exist_ok=True)

        if image_source == "Upload an image":
            img_path = os.path.join(input_dir, "uploaded_image.jpg")
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            img_path = uploaded_file  # For sample images, use the full path

        # Process the image
        with st.spinner("Processing image..."):
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            restored_img = inference(args.model_path, img_path, args, params, device, apply_dehazing=dehaze_image, conf_threshold=confidence_threshold)
            
            progress_placeholder.empty()

        output_path = os.path.join(output_dir, "processed_image.jpg")
        cv2.imwrite(output_path, restored_img)
        
        # Display images side by side
        col5, col6 = st.columns(2)
        
        image_width = 250  # Adjust as needed
        
        with col5:
            st.subheader("Original Image")
            st.image(Image.open(img_path), caption='Original Image', width=image_width)

        with col6:
            caption = 'Dehazed Image with Detections' if dehaze_image else 'Image with Detections'
            st.subheader(caption)
            st.image(Image.open(output_path), caption=caption, width=image_width)
        
        action = "restored and detected objects in" if dehaze_image else "detected objects in"
        st.success(f"Successfully {action} the image!")

        _, download_col, _ = st.columns([1,2,1])
        with download_col:
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

    # Tips section
    st.markdown("---")
    st.subheader("Tips for best results:")
    tips_col1, tips_col2, tips_col3 = st.columns(3)
    with tips_col1:
        st.markdown("- Use high-resolution images for better detection")
    with tips_col2:
        st.markdown("- Adjust confidence threshold for detection sensitivity")
    with tips_col3:
        st.markdown("- Compare results with and without dehazing")
        
def add_footer():
    st.markdown("---")
    st.markdown("Created by [Lim Xin/Multimedia University (MMU)]. For more information, visit [https://github.com/].")

def main():
    st.set_page_config(layout="wide", page_title="Mix-YoloNet: Dehazing and Object Detection")
    
    args, params = load_args_params()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tab1, tab2, tab3, tab4 = st.tabs(["Home", "Results Comparison", "Qualitative Evaluation", "Inference"])

    with tab1:
        home_page()
    with tab2:
        results_comparison_page()
    with tab3:
        qualitative_evaluation_page()
    with tab4:
        inference_page(args, params, device)

    add_footer()

if __name__ == "__main__":
    main()