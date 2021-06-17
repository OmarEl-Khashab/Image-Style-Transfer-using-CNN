# class Solution:
#     def reverseString(self, s):
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         string_postion = len(s) - 1
#         reverse_string = " "
#         while string_postion >= 0:
#             reverse_string +=s[string_postion]
#             string_postion -= 1
#
#         return reverse_string

class Solution:
    def reverseString(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left, right = left + 1, right - 1
call = Solution()
print(call.reverseString(["h","e","l","l","o"]))

