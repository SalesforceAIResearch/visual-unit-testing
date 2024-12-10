"""

Adapted from: https://github.com/cvlab-columbia/viper
"""

from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from viper_utils import show_single_image, load_json
from vision_processes import forward, viper_config

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(object_name, property: str)->bool 
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None, category=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) < 3:
                image = np.stack([image, image, image], axis=-1)
                image = torch.tensor(image)
            elif image.shape[0] == 1:
                image = np.stack([image[0], image[0], image[0]], axis=-1)
                image = torch.tensor(image)
            elif image.shape[0] != 3:   
                image = torch.tensor(image).permute(1, 2, 0)
            else:
                image = torch.tensor(image)
        if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255
        self.image = image

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            # self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]
        self.category = category

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2
        
        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('useful_lists/possible_options.json')

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image
    
    @property
    def area(self):
        """
        Returns the area of the bounding box.

        Examples
        --------
        >>> # What color is the largest foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patches.sort(key=lambda x: x.area)
        >>>     largest_foo_patch = foo_patches[-1]
        >>>     return largest_foo_patch.simple_query("What is the color?")
        """
        return self.width * self.height

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        # if object_name in ["object", "objects"]:
        #     all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
        # else:
        if viper_config.detect_model_name == 'dino':
            all_object_coordinates = self.forward('dino', self.cropped_image, object_name)
        else:
            all_object_coordinates = self.forward('dino_hf', image=self.cropped_image, obj=object_name)

        if len(all_object_coordinates) == 0:
            return []

        threshold = viper_config.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]
        # from pdb import set_trace; set_trace()
        results = []
        for coordinates in all_object_coordinates:
            try:
                results.append(self.crop(*coordinates))
            except Exception as e:
                print(e)
                print("Coordinates:", coordinates)
                continue
        return results[:10]

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def _score(self, category: str, negative_categories=None, model='clip') -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        if model == 'clip':
            res = self.forward('clip', self.cropped_image, category, task='score',
                               negative_categories=negative_categories)
        elif model == 'tcl':
            res = self.forward('tcl', self.cropped_image, category, task='score')
        else:  # xvlm
            task = 'binary_score' if negative_categories is not None else 'score'
            res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
            res = res.item()

        return res

    def _detect(self, category: str, thresh, negative_categories=None, model='clip') -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = viper_config.verify_property.model
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        if model == 'clip':
            return self._detect(name, negative_categories=negative_categories,
                                thresh=viper_config.verify_property.thresh_clip, model='clip')
        elif model == 'tcl':
            return self._detect(name, thresh=viper_config.verify_property.thresh_tcl, model='tcl')
        else:  # 'xvlm'
            return self._detect(name, negative_categories=negative_categories,
                                thresh=viper_config.verify_property.thresh_xvlm, model='xvlm')

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = viper_config.best_match_model
        image = self.cropped_image
        text = option_list_to_use
        if model_name in ('clip', 'tcl'):
            selected = self.forward(model_name, image, text, task='classify')
        elif model_name == 'xvlm':
            res = self.forward(model_name, image, text, task='score')
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        return option_list[selected]

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        if viper_config.simple_query_model == 'internlm':
            return self.forward('internlm', self.cropped_image, question, task='qa')
        else:
            return self.forward('blip', self.cropped_image, question, task='qa')
        
    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
                              self.left:self.right]
        return depth_map.median()  # Ideally some kind of mode, but median is good enough for now
    
    
    
    def from_bounding_box(cls, image, bounding_box):
        """Initializes an ImagePatch object by cropping the image at the given
        coordinates and stores the coordinates as attributes.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        bounding_box : dict
            A dictionary like {"box": [left, lower, right, upper], "category": str}."""
        left, lower, right, upper = bounding_box["box"]
        return ImagePatch(image, left, lower, right, upper, cls.left, cls.lower, queues=cls.queues, parent_img_patch=cls, category=bounding_box['category'])
    

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if viper_config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)
    
    def crop_left_of_bbox(self, left, lower, right, upper):
        """Returns an ImagePatch object representing the area to the left of the given
        bounding box coordinates.

        Parameters
        ----------
        left, lower, right, upper : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar to the left of the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     left_of_foo_patch = image_patch.crop_left_of_bbox(
        >>>         foo_patch.left, foo_patch.lower, foo_patch.right, foo_patch.upper
        >>>     )
        >>>     return bool_to_yesno(left_of_foo_patch.verify_property("bar", "quuxy"))
        """
        left_area = (0, self.lower, left, self.upper)
        return self.crop(*left_area)
    
    def crop_right_of_bbox(self, left, lower, right, upper):
        """Returns an ImagePatch object representing the area to the right of the given
        bounding box coordinates.

        Parameters
        ----------
        left, lower, right, upper : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar to the right of the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     right_of_foo_patch = image_patch.crop_right_of_bbox(
        >>>         foo_patch.left, foo_patch.lower, foo_patch.right, foo_patch.upper
        >>>     )
        >>>     return bool_to_yesno(right_of_foo_patch.verify_property("bar", "quuxy"))
        """
        right_area = (right, self.lower, self.width, self.upper)
        return self.crop(*right_area)

    def crop_below_bbox(self, left, lower, right, upper):
        """Returns an ImagePatch object representing the area below the given
        bounding box coordinates.

        Parameters
        ----------
        left, lower, right, upper : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar below the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     below_foo_patch = image_patch.crop_below_bbox(
        >>>         foo_patch.left, foo_patch.lower, foo_patch.right, foo_patch.upper
        >>>     )
        >>>     return bool_to_yesno(below_foo_patch.verify_property("bar", "quuxy"))
        """
        below_area = (self.left, self.lower, self.right, lower)
        return self.crop(*below_area)
    
    def crop_above_bbox(self, left, lower, right, upper):
        """Returns an ImagePatch object representing the area above the given
        bounding box coordinates.

        Parameters
        ----------
        left, lower, right, upper : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar above the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     above_foo_patch = image_patch.crop_above_bbox(
        >>>         foo_patch.left, foo_patch.lower, foo_patch.right, foo_patch.upper
        >>>     )
        >>>     return bool_to_yesno(above_foo_patch.verify_property("bar", "quuxy"))
        """
        above_area = (self.left, upper, self.right, self.upper)
        return self.crop(*above_area)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = viper_config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if viper_config.llm_source == 'hf':
        if long_answer:
            return forward(model_name='llm_general', prompt=query, queues=queues)
        else:
            return forward(model_name='llm_qa', prompt=[query, context], queues=queues)
    if long_answer:
        return forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)


def process_guesses(prompt, guess1=None, guess2=None, queues=None):
    if viper_config.llm_source == 'hf':
        return forward(model_name='llm_guess', prompt=prompt, queues=queues)
    return forward(model_name='gpt3_guess', prompt=[prompt, guess1, guess2], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric
