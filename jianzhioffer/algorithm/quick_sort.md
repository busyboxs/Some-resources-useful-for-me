# 快速排序

[online ide test here]https://ide.geeksforgeeks.org/haPPNbhU0L)

```cpp
#include <iostream>
#include <vector>
using namespace std;

void quickSort(vector<int>& nums, int l, int r);
int partition(vector<int>& input, int l, int r);
void print_vector(vector<int> nums);

void quickSort(vector<int>& nums, int l, int r) {
    if(l < r){
        int p = partition(nums, l, r);
        quickSort(nums, l, p-1);
        quickSort(nums, p+1, r);
    }
}

int partition(vector<int>& input, int l, int r) {
        int key = input[l];
        int i = l+1;
        int j = r;
        while(true){
            while(i < r && input[i] < key)
                i++;
            while(l < j && input[j] > key)
                j--;
            if(i>=j) {
                break;
            }
            swap(input[i], input[j]);
        }
        swap(input[j], input[l]);
        return j;
    }

void print_vector(vector<int> nums) {
    for(auto num: nums) {
	    cout<<num<<" ";
	}
	cout<<endl;
}

int main() {
	vector<int> nums = {3, 5, 7, 1, 2, 4, 6};
	quickSort(nums, 0, nums.size()-1);
	//partition(nums, 0, nums.size()-1);
	print_vector(nums);
	return 0;
}
```
