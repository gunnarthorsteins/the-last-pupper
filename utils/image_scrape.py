"""
This file contains functions useful for scraping and resizing internet images.

Alex Angus

November 11, 2019 (modified March 2021 for Deep Learning Project)
"""
import requests
from bs4 import BeautifulSoup
import sys
import numpy as np
import os

# dictionary storing artist name syntax (key) to identify artists and the link to all their works on WikiArt (value)
PAINTER_DICT = {'pablo-picasso' : 'https://www.wikiart.org/en/pablo-picasso/all-works/text-list',
                'william-turner' : 'https://www.wikiart.org/en/william-turner/all-works/text-list',
                'pierre-auguste-renoir' : 'https://www.wikiart.org/en/pierre-auguste-renoir/all-works/text-list',
                'vincent-van-gogh' : 'https://www.wikiart.org/en/vincent-van-gogh/all-works/text-list',
                'paul-cezanne' : 'https://www.wikiart.org/en/paul-cezanne/all-works/text-list',
                'claude-monet' : 'https://www.wikiart.org/en/claude-monet/all-works/text-list',
                'edouard-manet' : 'https://www.wikiart.org/en/edouard-manet/all-works/text-list',
                'jacques-louis-david' : 'https://www.wikiart.org/en/jacques-louis-david/all-works/text-list',
                'gustave-courbet' : 'https://www.wikiart.org/en/gustave-courbet/all-works/text-list',
                'eugene-delacroix' : 'https://www.wikiart.org/en/eugene-delacroix/all-works/text-list',
                'rembrandt' : 'https://www.wikiart.org/en/rembrandt/all-works/text-list',
                }


def download_image(url, filename):
    """
    downloads the image at the url to a file specified by filename
    """
    r = requests.get(url, stream=True)              #request page as stream
    if r.status_code == 200:                        #check status of request
        with open(filename, 'wb') as f:             #create image file
            for chunk in r.iter_content(1024):      #write image to file in chunks
                f.write(chunk)
    else:
        print("Request was unsuccessful!")
        sys.exit()


def get_image_urls(store=True, print_links=False, num_links=None, artists=PAINTER_DICT.keys()):
    """
    This function navigates through the wiki-art website starting at the page
    of each artist. It returns a list of strings specifying the url to
    an image of each of that artist's paintings.

    params:
        store: boolean specifying if painting urls should be saved (True) or not (False)
        print_links: print links to terminal as they're being stored
        num_links: the number of links to store per artist (default is None which means all possible links)

    returns:
        painting_links: a dictionary where keys are identical to PAINTER_DICT and
        values are a list of urls of painting images by the artist specified in keys
    """
    if 'data' not in os.listdir():                                              # generate data folder
        os.mkdir('data')
    if 'painting_links' not in os.listdir('data'):                              # generate painting links folder
        os.mkdir('data/painting_links')
    painting_links = {}
    for artist in artists:                                                      # for each artist
        print("Retrieving links for: {}".format(artist))
        r = requests.get(PAINTER_DICT[artist])                                  # request page listing all works
        painting_links.update({artist : []})
        if r.status_code == 200:                                                # if request is successful
            soup = BeautifulSoup(r.text, "html.parser")
            paintings = soup.find_all('li')                                     # get all links to paintings
            count = 0
            for class_ in paintings:                                            # filter non-painting links
                if (num_links is not None) and (count >= num_links):
                    break
                try:
                    link = class_.a.get('href')                                 # get download link
                except:
                    link = ''
                if artist in link:                                              # if the artist's name is in the link
                    image_url = get_image_link('https://www.wikiart.org' + link)# append domain and get link with get_image_link()
                    if image_url is not None:
                        count += 1
                        painting_links[artist].append(image_url)                # add download url to link list
                        if print_links:
                            print(image_url)

        if store:                                                               # save painting links
            links = [i for i in painting_links[artist] if i]
            np.savetxt('data/painting_links/' + artist, links, fmt='%s')

    return painting_links


def get_image_link(page_url):
    """
    returns the image url (string) of the image on the page specified by page_url
    """
    r = requests.get(page_url)                                                  # request page
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        try:
            image_link = soup.img.get('src')                                    # get link to image download
        except:
            image_link = None
    else:
        return None
    return image_link


def load_urls(np_file):
    """
    returns a numpy array stored in the file specified by np_file
    """
    return np.loadtxt(np_file, dtype=np.unicode_)


def store_images(artist, images=1e8):
    """
    downloads images in painting_links for artist
    """
    if 'data' not in os.listdir():                                                      # generate data folder
        os.mkdir('data')
    if 'painting_links' not in os.listdir('data'):                                      # generate painting links folder
        raise Exception('You need to call get_image_urls() before downloading images.')
    if 'images' not in os.listdir('data'):                                              # generate data folder
        os.mkdir('data/images')
    if artist not in os.listdir('data/images'):                                         # generate artist's images folder
        os.mkdir('data/images/{}'.format(artist))
    print("Downloading images for: {}".format(artist))
    filename = 'images/{}'.format(artist)
    artist_urls = load_urls('data/painting_links/{}'.format(artist))                    # get list of artist images download links
    count = 0
    for url in artist_urls:
        if count < images:
            download_image(url, 'data/images/{}/{}.jpg'.format(artist, str(count)))     # download images at download links
            count += 1


def download_data(artists=None, images_per_artist=1e8):
    """
    Download images from wikiArt. Combines the above functions into a single process.

    param:
        artists: list of artist names as in PAINTER_DICT.
                 If None, downloads images for all artists in PAINTER_DICT.

        images_per_artist: the number of images to download per artist
    """
    if artists is None:
        artists = PAINTER_DICT.keys()                                           # if artists is not specified, download images for all artists
    for artist in artists:
        store_images(artist, images=images_per_artist)
