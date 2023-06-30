import React from 'react'
import { useEffect, useState } from 'react';
import { Pagination, PaginationItem, PaginationLink } from 'reactstrap';

export default function PaginationComponent({ PageIndex, totalPage, handlePageChange }) {
    var startPage, endPage;
    if (totalPage <= 5) {
        startPage = 1;
        endPage = totalPage;
    } else {
        if (PageIndex <= 3) {
            startPage = 1;
            endPage = 5;
        } else if (PageIndex + 2 >= totalPage) {
            console.log(PageIndex + 2);
            startPage = totalPage - 5;
            endPage = totalPage;
        } else {
            startPage = PageIndex - 2;
            endPage = PageIndex + 3;
        }
    }
    
    var pages = [...Array((endPage + 1) - startPage).keys()].map(i => startPage + i);
    return (
        <Pagination size="sm">
            <PaginationItem disabled>
                <PaginationLink>
                    Trang {PageIndex}/{totalPage}
                </PaginationLink>
            </PaginationItem>
            <PaginationItem disabled={PageIndex == 1}>
                <PaginationLink onClick={() => handlePageChange(1)}>
                    First
                </PaginationLink>
            </PaginationItem>
            <PaginationItem disabled={PageIndex == 1}>
                <PaginationLink previous onClick={() => handlePageChange(PageIndex - 1)} />
            </PaginationItem>
            {
                pages.map((item) => (
                    <PaginationItem key={item} active={PageIndex == item}>
                        <PaginationLink onClick={() => handlePageChange(item)}>
                            {item}
                        </PaginationLink>
                    </PaginationItem>
                ))
            }
            <PaginationItem disabled={PageIndex == totalPage}>
                <PaginationLink next onClick={() => handlePageChange(PageIndex + 1)} />
            </PaginationItem>
            <PaginationItem disabled={PageIndex == totalPage}>
                <PaginationLink onClick={() => handlePageChange(totalPage)}>
                    Last
                </PaginationLink>
            </PaginationItem>
        </Pagination>
    )
}
