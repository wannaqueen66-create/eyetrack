# QC exclusion list

Current after-QC rerun excludes these 8 participants and reruns the mainline pipeline from filtered input, rather than only hiding them in figures.

- 孙校聪
- 康少勇
- 张钰鹏
- 杨可
- 洪婷婷
- 陈韬
- 高梓楠
- 赵国宏

Source config:
- `configs/excluded_participants_qc.csv`

Mainline output tracks:
- `00_全样本_AllSample`
- `01_QC后_AfterQC`
