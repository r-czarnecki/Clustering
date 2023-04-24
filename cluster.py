from PIL import Image
import numpy as np
import math
import statistics as stat
import datetime as dt
import cv2
import threading
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import sys
import ntpath

def read_im(file, add_pref=True):
    if add_pref:
        path = './training_sample_ocr/probka_uczaca_zad2/' + file
    else:
        path = file
    im = Image.open(path).convert('RGB')
    im = np.array(im)

    return im

def sim(im1, images, draw=False, match_size=False, threshold=255):
    ratio_penalty_mult = 1
    im1 = [im1 for i in range(0, len(images))]
    if match_size:
        r1 = [len(i[0]) / len(i) for i in im1]
        r2 = [len(im2[0]) / len(im2) for im2 in images]
        ratio_penalty_mult = np.max([r1, r2], 0) / np.min([r1, r2], 0)
        im1_heights = [len(i) for i in im1]
        im2_heights = [len(im2) for im2 in images]
        new_height = np.max([im1_heights, im2_heights], 0)
        
        im1_widths = [len(i[0]) for i in im1]
        im2_widths = [len(im2[0]) for im2 in images]
        new_width = np.max([im1_widths, im2_widths], 0)
        im1 = resizeMultiple(im1, new_height, new_width)
        images = resizeMultiple(images, new_height, new_width)
    
    diff = [blur(np.abs(i1.astype('int') - i2.astype('int'))) for i1,i2 in zip(im1, images)]
    diff = [d**2 for d in diff]
    
    sim = np.array([np.sum(i) / 3 for i in diff])
    size_reward = np.array([(len(i) * len(i[0])) for i in im1])**0.5
    ratio_penalty_mult[ratio_penalty_mult>1.5] += 1000
    return (sim * ratio_penalty_mult * 100 / size_reward)**0.5

def resizeMultiple(images, height, width):
    return [cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_NEAREST) for im,h,w in zip(images, height, width)]

def crop(im, threshold):
    left = len(im[0])
    right = 0
    top = len(im)
    bottom = 0
    for i, row in enumerate(im):
        if np.min(row) <= threshold:
            top = min(i, top)
            bottom = max(i, bottom)

    new_im = np.transpose(im, [1, 0, 2])
    for j, column in enumerate(new_im):
        if np.min(column) <= threshold:
            left = min(j, left)
            right = max(j, right)

    new_im = new_im[left:right + 1]
    new_im = np.transpose(new_im, [1, 0, 2])
    new_im = new_im[top:bottom + 1]

    return new_im

def load_all(file):
    images = []
    paths = []
    with open(file, "r") as f:
        img = f.readline()

        while img:
            img = img.strip()
            paths.append(img)
            images.append(read_im(img, add_pref=False))
            img = f.readline()

    return images, paths

def sim_with_cropping(im1, images, cr_val, draw=False, step=25):
    cr1 = crop(im1, cr_val)
    cropped = [crop(im2, cr_val) for im2 in images]
    return sim(cr1, cropped, match_size=True, draw=draw, threshold=cr_val)

def compute_sim(im1, images, cr_val, draw=False, step=25):
    im1 = copy.deepcopy(im1)
    images = copy.deepcopy(images)
    min_val = max(np.min(im1), max([np.min(im) for im in images]))
    sim_rates = [sim_with_cropping(im1, images, i, draw=draw, step=step) for i in range(min_val, 256, step)]
    sim_rates = np.array(sim_rates)
    sim_rates = np.sort(sim_rates, 0)
    filtered = sim_rates[int(len(sim_rates) / 4):int(len(sim_rates) * 3 / 4)]
    return np.mean(filtered, 0)

def center(images):
    max_width = np.max([len(img[0]) for img in images])
    avg_ratio = np.sum([len(img[0]) / len(img) for img in images])

    avg_ratio /= len(images)
    height = round(max_width / avg_ratio)

    center_img = np.zeros((height, max_width, 3))
    resized_images = np.array(resizeMultiple(images, [height for i in np.arange(len(images))], [max_width for i in np.arange(len(images))]))
    center_img = np.sum(resized_images, 0)

    return np.around(center_img / len(images)).astype('int')

def blur(image):
    return np.round(cv2.GaussianBlur(image.astype('float32'), (3,3), 0)).astype('int')
    
def worker(start, step, ungroupped, clusters, distance, input_queue, output_queue):
    i = start
    while i < len(clusters):
        try:
            while True:
                message = input_queue.get_nowait()
                if message[0] == 'UPDATE_CLUSTER':
                    clusters[message[1]][0] = message[2]
                elif message[0] == 'NEW_UNGROUPPED':
                    ungroupped = message[1]
                elif message == 'END':
                    return
        except:
            pass

        img = clusters[i][1]
        tmp = [image[0] for image in ungroupped]
        mean = compute_sim(img, tmp, 20, step=50, draw=False)

        res1 = []
        res2 = []
        for image, dist in zip(ungroupped, mean):
            if dist <= distance:
                res1.append(image)
                res2.append(dist)

        output_queue.put((i, res1, res2))
        i += step

    while True:
        try:
            message = input_queue.get_nowait()
            if message == 'END':
                return
        except:
            pass

def compute_clusters(images, distance):
    clusters = [[0, image] for image in images]
    ungroupped = [(image, i) for i, image in enumerate(images)]
    finished = {}
    clusters_dict = {}
    last_cluster = 0
    clustered_images = 0
    process_number = 5
    processes = []
    queues = []
    input_queue = Queue()

    print("[{}] Starting processes...".format(dt.datetime.now().strftime("%H:%M:%S")))
    for i in range(0, process_number):
        queues.append(Queue())
        processes.append(Process(target=worker, args=(i, process_number, ungroupped, clusters, distance, queues[-1], input_queue)))
        processes[-1].start()

    for i, img in enumerate(images):
        if clusters[i][0] == 0:
            last_cluster += 1
            clusters_dict[last_cluster] = []
            tmp_dict = copy.deepcopy(clusters_dict)
            tmp_dict[last_cluster] = [img]
            clusters[i][0] = last_cluster
            for q in queues:
                q.put(("UPDATE_CLUSTER", i, last_cluster))

            while i not in finished:
                message = input_queue.get()
                finished[message[0]] = (message[1], message[2])

            local_ungroupped = finished[i][0]
            local_mean = finished[i][1]

            clustered_images += 1
            for j, m in zip(local_ungroupped, local_mean):
                j = j[1]
                if m <= distance and clusters[j][0] == 0:
                    clustered_images += 1
                    clusters[j][0] = last_cluster
                    tmp_dict[last_cluster].append(clusters[j][1])
                    for q in queues:
                        q.put(("UPDATE_CLUSTER", j, last_cluster))

            ungroupped = []
            for x, j in enumerate(clusters):
                if j[0] != 0:
                    tmp_dict[j[0]].append(j[1])
                else:
                    ungroupped.append((j[1], x))

            for q in queues:
                q.put(("NEW_UNGROUPPED", ungroupped))

            cntr = [center(val) for val in tmp_dict.values()]
            distances = [(d, c, ctr) for d, c, ctr in zip(range(0, len(cntr)), tmp_dict.keys(), cntr)]
            my_center = center(tmp_dict[clusters[i][0]])
            center_distances = compute_sim(my_center, cntr, 20, step=50, draw=False)
            for closest in distances:
                if closest[1] == clusters[i][0]:
                    continue
                cntr_dist = center_distances[closest[0]]
                if cntr_dist <= distance:
                    tmp = closest[1]
                    tmp_dict[clusters[i][0]].extend(tmp_dict[closest[1]])
                    del tmp_dict[closest[1]]
                    del clusters_dict[closest[1]]
                    for im in clusters:
                        if im[0] == tmp:
                            im[0] = clusters[i][0]
                    my_center = center(tmp_dict[clusters[i][0]])
                    center_distances = compute_sim(my_center, cntr, 20, step=50, draw=False)
            
            print("[{}] Progress: {:.2f}% ({}/{}) | Number of clusters: {}".format(dt.datetime.now().strftime("%H:%M:%S"), clustered_images / len(images) * 100, clustered_images, len(images), len(clusters_dict.keys())))

    for t, q in zip(processes, queues):
        q.put(("END"))
        t.join()
        q.close()

    input_queue.close()

    for x, i in enumerate(clusters):
        clusters_dict[i[0]].append([i[1], x])
    return clusters_dict

print("[{}] Loading data...".format(dt.datetime.now().strftime("%H:%M:%S")))
all_images, all_paths = load_all(sys.argv[1])

cl = compute_clusters(all_images, 1700)

print("[{}] Generating HTML file...".format(dt.datetime.now().strftime("%H:%M:%S")))

site = ""
for i in cl:
    site += "<div>"
    for img in cl[i]:
        site += '<div style="\
                    display: inline-block;\
                    border: black;\
                    border-width: 10px;\
                    margin: 2px;\
                    border-style: double;\
                ">'
        site += "<img src='{}' height=100px width=auto>".format(all_paths[img[1]])
        site += "</div>"
    site += "</div><hr>"

with open("clusters.html", "w") as f:
    f.write(site)

print("[{}] generating cluster file...".format(dt.datetime.now().strftime("%H:%M:%S")))
string = ""
for i in cl:
    for img in cl[i]:
        string += ntpath.basename(all_paths[img[1]]) + " "
    string += "\n"

with open("clusters.txt", "w") as f:
    f.write(string)