# Aspect_Sent_project

Выполнили: *Юлия Короткова и Яков Раскинд*

### Запуск скрипта на инференс

Файл для инференса находится в [`inference.py`](https://github.com/yuliya1324/Aspect_Sent_project/blob/main/inference.py)

```
python inference.py --first_task --second_task --aspect_filename data/dev_pred_aspects.txt --sentiment_filename data/dev_pred_cats.txt --reviews_filename data/dev_reviews.txt --model_cat <path_to_model_cat> --model_sent <path_to_model_sent> --model <path_to_model>
```

Аргументы `--first_task` `--second_task` задают, какое задание надо выполнить (1. аспекты, 2. сентименты)

Остальные аргументы передают пути к файлам и моделям. `model_cat` -- Модель для выделения аспектов по категориям. `model_sent` -- Модель для определения сентимента аспектов. `model` -- Модель для классификации сентиментов по категориям.

Ссылки для скачивания моделей:

[Модель для классификации сентиментов по категориям](https://disk.yandex.ru/d/bUbzbGnFqNywHw)

[Модель для выделения аспектов по категориям](https://disk.yandex.ru/d/dStzQfoH_33Opw)

[Модель для определения сентимента аспектов](https://disk.yandex.ru/d/O0MOfzkp0ownag)

### Отчет

Отчет находится в [`report.ipynb`](https://github.com/yuliya1324/Aspect_Sent_project/blob/main/report.ipynb)
