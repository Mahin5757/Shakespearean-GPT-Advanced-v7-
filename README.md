Shakespearean GPT (Advanced v7)
This project contains the implementation of "Shakespearean GPT," an advanced, high-performance autoregressive language model based on the GPT architecture. It is specifically designed and trained to generate high-quality text in the style of William Shakespeare. The model incorporates modern deep learning techniques such as Grouped-Query Attention (GQA), RoPE positional embeddings, RMSNorm, and SwiGLU activations for enhanced performance and parameter efficiency.

Beyond simple text generation, this project includes a fully functional, interactive chatbot that allows users to converse with the "Bard" in real-time.

Key Features
Modern Transformer Architecture: Built with cutting-edge components for optimal performance:

Grouped-Query Attention (GQA): Balances the computational efficiency of Multi-Query Attention with the quality of Multi-Head Attention.

Rotary Positional Embeddings (RoPE): A sophisticated method for encoding token positions, leading to better performance on long sequences.

RMSNorm: A stable and efficient normalization technique used throughout the model.

SwiGLU FFN: A variant of the Swish activation function in the feed-forward network, which often improves performance.

Custom BPE Tokenizer: Comes with a Byte-Pair Encoding (BPE) tokenizer trained specifically on a cleaned Shakespearean dataset. The tokenizer can be easily retrained on new or modified data.

Advanced Training & Fine-Tuning:

Support for fine-tuning from existing checkpoints.

Gradient Accumulation to simulate larger batch sizes on memory-constrained hardware.

Cosine learning rate schedule with warmup for stable convergence.

AdamW optimizer with weight decay for better regularization.

Gradient Clipping to prevent exploding gradients.

Automatic Mixed Precision (AMP) for faster training on supported NVIDIA GPUs.

Gradient Checkpointing to reduce memory usage during training, allowing for larger models.

Sophisticated Text Generation:

Supports standard sampling methods like temperature, top-k, and top-p (nucleus) sampling.

Includes an implementation of Mirostat sampling, an advanced technique that controls the perplexity of the generated text for more coherent and less repetitive output.

Repetition penalty to discourage the model from repeating itself.

Interactive Chatbot: Engage in a conversation with the trained model. The chatbot maintains a conversation history and is designed to respond to modern English with Shakespearean flair.

PyTorch torch.compile Integration: The script is designed to be compatible with torch.compile for significant speedups during training and inference, with graceful fallbacks for environments where it is not supported.

Requirements
The primary requirements for this project are listed in the requirements.txt file. The core dependencies are:

torch

tokenizers (from Hugging Face)

requests

To install all dependencies, run:

pip install -r requirements.txt

How to Use
1. Download the Dataset
The script will automatically download the Shakespearean text dataset upon first run. If you wish to use your own cleaned dataset, simply update the DATA_PATH variable in the script and ensure the file exists.

2. Train the Model
To train the model from scratch or continue training from a checkpoint, run the script and select option 1:

python your_script_name.py

If a MODEL_SAVE_PATH file is found, you will be prompted to load it and continue training.

If you wish to start fresh, either delete the existing model checkpoint or select "no" when prompted.

Training progress, including validation loss and perplexity, will be printed to the console. The best model checkpoint will be saved to the path specified in MODEL_SAVE_PATH.

3. Interact with the Chatbot
Once a model is trained and a checkpoint file exists, you can run the interactive chatbot. Run the script and select option 2:

python your_script_name.py

You can then chat with the model. Type new to start a new conversation or exit to quit.

4. Generate Sample Text
To generate a block of text from a prompt, run the script and select option 3:

python your_script_name.py

You will be prompted to enter a starting phrase. If you leave it blank, the model will start generating from the beginning-of-sequence token.

Configuration
All major hyperparameters for the model, training, and generation can be configured in the top section of the Python script. This includes:

Model dimensions (n_embd, n_layer, n_head, etc.)

Training parameters (BATCH_SIZE, MAX_LEARNING_RATE, MAX_ITERS, etc.)

File paths (DATA_PATH, TOKENIZER_JSON_PATH, MODEL_SAVE_PATH)

Generation settings (GENERATION_TEMPERATURE, GENERATION_TOP_K, etc.)

Feel free to experiment with these settings to train different model variants or to achieve different generation styles.
