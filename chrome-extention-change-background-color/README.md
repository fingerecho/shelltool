参考https://blog.csdn.net/fyyyr/article/details/80949983


manifest.json说明
1.    必须字段是必填的。

2.    建议字段可选填。注意icons数组，与添加的png一一对应。

3.    由于要在地址栏右侧显示图标，所以可选字段使用browser_action。不希望点击图标弹出气泡页面，所以不设置其default_popup属性。

4.    由于只需要一个js注入功能，所以自定义字段只需要一个content_scripts，permissions在这里可以省略。content_scripts需要设置mathces属性来匹配百度的页面，然后指定要注入的js。因为要引用jquery库，所以先注入jquery，再注入自定义的content.js。

