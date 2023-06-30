import { toast } from 'react-toastify';

export const toastSuccess = () => {
    toast.success("Thành công");
}
export const toastDeleteSuccess = () => {
    toast.success("Đã xóa");
}
export const toastError = (error) => {
    let message = null;

    if (typeof error === 'object' && error.message) {
        ({ message } = error);
    }
    if (message !== null && message !== undefined) {
        toast.error(message);
    }
}


export const toastSuccessText = (txt) => {
    toast.success(txt);
}
export const toastErrorText = (txt) => {
    toast.error(txt);
}