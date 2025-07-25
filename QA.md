# SpectraGAN: FAQs

## 1. Why this problem?

High-resolution satellite imagery can be used in urban planning, agriculture, and disaster management. However, acquiring high-resolution data from satellite platforms is an expensive operation due to resource constraints. Hence, satellites acquire multiple low-resolution images often shifted by half a pixel in both along and across track directions. These low-resolution images are utilized to generate high-resolution images. We want to provide super resolution models apt for low res- satellite imagery, that can capture high frequency changes of aerial images.

## What is blind-assessment?

The quality of high-resolution images often depends on various satellite system parameters including the algorithm used for super-resolving. However, assessing the quality of these generated super-resolved images is challenging due to the absence of ground-truth references. This also necessitates the use of blind (no-reference) quality assessment techniques that can evaluate both the perceptual realism and fidelity of super-resolved images.

## 2. What dataset was used?

We used the [SpectraGAN_DATA dataset](https://github.com/ParamAhuja/SpectraGAN), which contains thousands of paired low- res images fom sentinel2 and high-res images from naip. This dataset is specifically curated for satellite image super-resolution tasks, making it ideal for both training and evaluating our model.

## 3. Why did you choose ESRGAN as your model architecture?

ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) is a state-of-the-art model for image super-resolution. It is known for producing perceptually realistic high-resolution images, which is essential for satellite imagery where fine details matter. Its architecture is also flexible for fine-tuning on domain-specific data.

## 4. How did you fine-tune your model?

We started with a pre-trained ESRGAN model and fine-tuned it on the SpectraGAN_DATA dataset. We used data augmentation, perceptual loss, and adversarial training to ensure the model could generalize well to satellite images and preserve important features.

## 5. How do you assess the quality of super-resolved images without ground-truth references?

We implemented a blind (no-reference) image quality assessment module that uses deep features to predict perceptual quality scores. This allows us to evaluate the realism and fidelity of super-resolved images even when ground-truth high-resolution images are unavailable.

## 6. How is the model integrated into the app?

The backend hosts the fine-tuned ESRGAN model and the quality assessment module. The frontend allows users to upload images, run super-resolution, and view quality scores. The backend processes images and returns both the enhanced image and its quality score to the frontend for display.

## 7. How can users deploy or test your solution?

- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Rockerleo/esrgan)
- **Local Deployment:** Follow the instructions in the README to set up the backend and frontend, install dependencies, and run the app locally.

## 8. What are the main limitations of your current solution?

- The blind quality assessment is trained on available data and may not generalize perfectly to all satellite image types or sensors.
- The model requires a GPU for fast inference, which may limit deployment on low-resource devices.
- The current UI is functional but could be further improved for usability and visualization.

## 9. What are the potential future improvements?

- Incorporate more diverse satellite datasets to improve generalization.
- Enhance the blind quality assessment module with additional perceptual metrics.
- Add support for batch processing and API endpoints for integration with other platforms.
- Improve the frontend for better user experience and visualization tools.

## 10. How does your solution impact real-world applications?

By enabling reliable, no-reference quality assessment of super-resolved satellite images, our solution empowers researchers, analysts, and organizations to make better use of low-cost satellite data, accelerating applications in environmental monitoring, urban planning, and beyond.

---

For any further questions, please refer to the [README](./README.md) or contact the project team.
