# gya-image-comparison



## Objectives

- develop a high-performance FastAPI application capable of comparing images
- allow users to upload two images that are to be compared
- result should indicate whether the two images are "matching" or "not matching"
- it should be able to tell whether the two images are a match regardless of whether or not they're of the same size



## Setup and Usage

Install the requirements

`pip install -r requirements.txt`

Run the application

`uvicorn script_name:app --reload`

Test the endpoint using curl

`curl -X POST http://127.0.0.1:8000/compare-images -F files=@img/0_source.jpg -F files=@img/0_comp_diff.jpg`



## Method
1. Identify the smaller and larger of the two images

2. Extract Scale Invariant keypoints from both images

3. Match keypoints of both images.

   ![](./img/keypoints.png)

4. Using keypoints warp the large image over the smaller one.

5. Compute the structural similarity indexes of the small and warped images

6. If the similarity is higher than 0.9 label as 'match'

   

## Tests

![output_0](./img/output_0.png)

2. Images of the same product but different resolutions

![output_1](./img/output_1.png)

3. Exact same images with just two characters missing

![output_2](./img/output_2.png)

4. Same item; different resolution

![output_3](./img/output_3.png)

5. Same item; different resolution

![output_4](./img/output_4.png)

6. Different items

![output_5](./img/output_5.png)

7. Same image; original and compressed

![output_6](./img/output_6.png)

## References

1. Lowe, David G. "Distinctive image features from scale-invariant keypoints." [[pdf](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)]
