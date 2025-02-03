void trans(x,y) {
    if(n==32) {
        transpose()
    }
    
    for(i = mid to down) {
        for j = left to mid {
            swap mat[i][j],mat[j][i]
        }
    }
}

void transpose_recursive(Matrix &result,int i,int j,int n) {
    if(n==32) {
        int ii=i,jj=j;
        for(;i<ii+n;+=i) {
            for(;j<jj+n;++j) {
                if(i<j) {
                    std::swap(result[i][j],result[j][i]);
                }
            }
        }
    }
    int mid = (i+n)/2
    transpose_recursive(result,i,j,n/2);
    transpose_recursive(result,mid,j,n/2);
    transpose_recursive(result,i,mid,n/2);
    transpose_recursive(result,mid,mid,n/2);

    for(;i<mid;++i) {
        for(j=mid;j<n;++j) {
            std::swap(result[i][j],result[mid+i][j-mid]);
        }
    }
}