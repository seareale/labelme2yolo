# labelme2yolo
Format converter from Labelme to YOLO

## Run
Before you start,
```
pip install -r requirements.txt
```

You can change dataset directory and labels in `conifg.yaml`.

```bash
python lableme2yolo.py --data config.yaml
```

## Results
The result of the conversion will be created in `./results` dir.  
(image : `images/{dir_name}`, label : `labels/{dir_name}`)

When the files that have a error is appeared, the files will be moved in `./results/{dir_name}/error` dir.  
