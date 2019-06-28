# Example

This is a tool that tries to identify an image that you give the system.

## Prerequesits

You're going to need NodeJS on your system in order for this to run.

## Setup

From the `examples` folder please run `yarn` or `npm install`

## Usage

This is a CLI tool and there are alreadys some example images that you can play around with in the `img` folder.

To get started run the following from `examples/js`:

```bash
> node mobilenet.js ../img/panda.jpg
```

Press `ctrl-c` to end the program.

You can also pass in multiple images and the app will build up a table of guesses.

```bash

> node mobilenet.js ../img/sample_computer.jpg ../img/panda.jpg ../img sample_dog.jpg

```
