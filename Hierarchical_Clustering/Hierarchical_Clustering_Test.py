'''
Created on May 4, 2017

@author: Leo Zhong
'''
import os
from PIL import Image, ImageDraw
from numpy import * 
import numpy as np
from nbformat.v2.tests.nbexamples import jpeg


def drawdendrogram(clust,imlist,jpeg="clusters.jpg"):
    #height and WIDTH
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)
    
    #width is fixed, scale distances accordingly
    scaling = float(w-150)/depth
    
    #create a new img with a white background
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    draw.line((0,h/2,10,h/2), fill=(255,0,0))
    
    #draw the first mode
    drawnode(draw,clust,10,int(h/2),scaling,imlist,img)
    img.save(jpeg)
    
def drawnode(draw,clust,x,y,scaling,imlist,img):
    if clust.id < 0:
        h1 = getheight(clust.left)*20
        h2 = getheight(clust.right)*20
        top = y - (h1+h2)/2
        bottom = y+(h1+h2)/2
        
        # line length
        l1 = clust.distance*scaling
        
        #vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        
        #horizontal line to left item
        draw.line((x,top+h1/2,x+11,top-h2/2),fill=(255,0,0))

        #horizontal line to left item
        draw.line((x,bottom+h1/2,x+11,bottom-h2/2),fill=(255,0,0))

        # call the function to draw the left and right node
        drawnode(draw,clust.left,x+11,top+h1/2,scaling,imlist,img)
        drawnode(draw,clust.right,x+11,bottom+h1/2,scaling,imlist,img)
        
    else:
        #if this the end node, draw a thumball img
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((20,20))
        ns = nodeim.size
        print(x,y-ns[1]//2)
        print(x+ns[0])


class cluster_node:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance
        self.count=count

def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))
    
def L1dist(v1,v2):
    return sum(abs(v1-v2))

def hcluster(features,distance):
    #cluster the rows of the "features" matrix
    distances={}
    currentclustid=-1

    # clusters are initially just the individual rows
    clust=[cluster_node(array(features[i]),id=i) for i in range(len(features))]

    while len(clust)>1:
        lowestpair=(0,1)
        closest=distance(clust[0].vec,clust[1].vec)
    
        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances: 
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)
        
                d=distances[(clust[i].id,clust[j].id)]
        
                if d<closest:
                    closest=d
                    lowestpair=(i,j)
        
        # calculate the average of the two clusters
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 \
            for i in range(len(clust[0].vec))]
        
        # create the new cluster
        newcluster=cluster_node(array(mergevec),left=clust[lowestpair[0]],
                             right=clust[lowestpair[1]],
                             distance=closest,id=currentclustid)
        
        # cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1
    
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0
    
    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance
      
#create a list od img
imlist = []
folderPath = r'F:\MachineLearningPractice\Img'
for filename in os.listdir(folderPath):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(os.path.join(folderPath,filename))
n = len(imlist)
print(n)

#extract feature vector for each img
features = np.zeros((n,3))
for i in range(n):
    im = np.array(Image.open(imlist[i]))
    R = np.mean(im[:,:,0].flatten())
    G = np.mean(im[:,:,1].flatten())
    B = np.mean(im[:,:,2].flatten())
    features[i]=np.array([R,G,B])
    
tree = hcluster(features)
drawdendrogram(tree, imlist, jpeg='hierarchical_clustering.jpg')


    
