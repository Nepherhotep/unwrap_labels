from setuptools import setup, find_packages


__author__ = 'Alexey Zankevich'


setup(
    name="unwrap_labels",
    version="1.1.3",
    py_modules=['unwrap_labels'],
    author=__author__,
    author_email="alex.zankevich@gmail.com",
    description="Library to unwrap labels using OpenCV",
    keywords=["OpenCV"],
    license="MIT",
    platforms=['Platform Independent'],
    url="https://github.com/Nepherhotep/unwrap_labels",
    install_requires=["numpy", "opencv-contrib-python"],
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10"]
)
