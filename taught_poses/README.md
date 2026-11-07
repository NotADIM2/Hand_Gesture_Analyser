# Taught poses

Each `.json` file here is one hand pose taught with the `t` key in the app. Every
file in this folder is loaded at start, so a pose committed here is recognised by
everyone who pulls the repository.

A file holds the pose's name, who taught it (`taught_by`), when, and the hand
landmarks of each recorded example. To remove a pose, delete its file.

To add one, see "Add a pose for everyone" in the main README. `pytest` checks every
file: valid format, a name that isn't a built-in gesture, and that it doesn't take
over any built-in gesture.
