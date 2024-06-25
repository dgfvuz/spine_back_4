from rest_framework.pagination import PageNumberPagination


class CustomPagination(PageNumberPagination):
    page_size = 10  # 每页显示10条记录
    page_query_param = 'p'  # 客户端用来指定页码的请求参数名改为'p'
    page_size_query_param = 'size'  # 允许客户端通过'size'参数来指定每页记录数
    max_page_size = 100  # 每页记录数的上限