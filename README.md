## MLMonkey

On my mac, create python 3 sandbox. Note, scikit-learn depends on numpy and scipy; matplotlib misses libpng-devel and libfreetype-devel on my mac. "brew install freetype" will take care both missing libraries.
```bash
/usr/local/bin/virtualenv -p /usr/local/bin/python3 ./mlmonkey
source ./mlmonkey/bin/activate
pip install numpy
pip install scipy
pip install scikit-learn
pip install pandas
brew install freetype
pip install matplotlib
```
