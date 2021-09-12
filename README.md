# Корупци-молка

## Подготовка среды запуска:
* Необходимо наличие установленной CUDA 10.2 SDK в системе, а также поддерживаемой видеокарты(4GB для обучения, ?GB для использования)
* Необходимо наличие conda в системе
* Все файлы запускаются из conda environment, указанного в environment.yml

### Чтобы пропустить этапы Обучение языковой модели и/или Обучение классификатора можно скачать архив с предобученными моделями по ссылке: ?

## Обучение языковой модели

Для старта обучения языковой модели достаточно запустить файл lm_trainer.py (выполнение может занять >12 часов)


## Обучение классификатора

(требует обученной языковой модели)

Для обучения модели нужно выполнить все ячейки `train.ipynb`, датасет будет загружен из папок соответствующих маске `DataSet_*`(не включены в репозиторий, `DataSet_Export` не используется).

## Использование модели

(требует обученного классификатора)

Для запуска сервера необходимо обучить или скачать предобученную модель(?), после этого достаточно выполнить команду `voila ui.ipynb --VoilaConfiguration.file_whitelist="['.*.docx']"`, сервис будет доступен на указном в терминале порту
