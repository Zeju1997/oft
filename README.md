# Controlling Text-to-Image Diffusion by Orthogonal Finetuning

## Description

Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.

Stay tuned, more information and code coming soon.

## Table of Contents

1. [Project Title](#project-title)
2. [Description](#description)
3. [Table of Contents](#table-of-contents)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

(Coming soon...) Here you'll explain how to install and setup your project.

## Usage 

(Coming soon...) Here you can show examples of how to use your project, preferably with screenshots or code snippets.

## Contributing

If you wish to contribute to this project, feel free to fork the repository and submit a pull request. We're always looking forward to improving this project and appreciate all the help we can get!

## License

Add information about your license here. If you're unsure what license to use, GitHub provides a [licensing guide](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository) that might help.

---

### Stay tuned for more updates!
