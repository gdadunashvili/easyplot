# easyplot

## what's the point of this

There are a large number of plotting frameworks in python, and it seems 
counterproductive to make a new one when the energy could be spent on working
 on existing ones. And I agree, there is probably a much better implemented 
 and maintained project which does everithyng this project does much better. 
 However I had a free weekend and wanted to 
 finally upload something on my github page so this is what I came up with.
 
This is a collection of tools developed during my master thesis to
make scientific plotting easier. The main idea is to take the huge 
functionality of matplotlibs pyplot and reduce it to one object with **very few, 
methods** implimenting **most often used functionality** and **sensible 
defaults** for scientific plotting. The driving idea I had in mind while 
working on this was the minimization of **ink to data ratio**.

## download and instalation

to download the project you can use git

```bash
git clone https://github.com/gdadunashvili/easyplot.git
cd easyplot 
```
after you have navigated into the folder you can use `pip` for installation

```bash
sudo pip install .
```
or alternatively

```bash
sudo python setup.py install
```
this module was written in and for `python3.6.1` and above and will not work 
for any older version. 
