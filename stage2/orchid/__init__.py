"""Minimal ORCHID comparative training package."""
import os

CLASS_NAMES = ["normal", "osmf", "oscc"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_ORDER = "normal=0|osmf=1|oscc=2"
