# LogicMate v3

## Disclaimer

This project asumes that your already have installed the following:

- Python 3.10.18

And you also have `CUDA` and `cuDNN` installed in your system if you want to use GPU.

## Initialization

There is a file `setup.sh` that will install all the dependencies for you. You can run it with the command:

```bash
./setup.sh
```

If you don't have permission to run the script, you can run the following command:

```bash
chmod +x setup.sh
```

Then run the script again.

To enter the virtual environment in your terminal, run:

```bash
source .venv/bin/activate
```

To exit the virtual environment, run:

```bash
deactivate
```

## Development

`Ruff` is present in the project to lint the code.

To run `Ruff`, you can use the following command:

Current file:

```bash
ruff check .
```

To check all files:

```bash
ruff check
```

To fix the code, you can use the following command:

Current file:

```bash
ruff format .
```

To check all files:

```bash
ruff format
```

For more information about `Ruff`, you can check the [Repository](https://github.com/astral-sh/ruff?tab=readme-ov-file).
