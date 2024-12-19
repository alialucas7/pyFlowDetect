
from setuptools import setup, find_packages

setup(
    name="pyflow-detect",  # Nombre del paquete
    version="0.1.1",  # Versión
    author="Alia Lucas Emanuel",
    author_email="lukitis2@gmail.com",
    description="Detect port scans in your network with python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alialucas7/ptFlowDetect",  # URL del repositorio
    packages=find_packages(),  # Busca automáticamente todos los módulos
    install_requires=open("requeriments.txt").read().splitlines(),  # Dependencias
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.10.12',  # Versión mínima de Python
)

