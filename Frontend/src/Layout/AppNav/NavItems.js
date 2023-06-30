export const Navigations = [
    {
        icon: 'pe-7s-mail',
        label: 'Quản lý công việc',
        content: [
            {
                label: 'Hồ sơ đến',
                to: '#/Dashboard/crm',
            },
            {
                label: 'CC',
                to: '#/task/cc',
            },
            {
                label: 'Hồ sơ gửi',
                to: '#/task/sendbox',
            },
            {
                label: 'Thảo luận',
                to: '#/task/discuss',
            },
            {
                label: 'Theo dõi',
                to: '#/task/follow',
            },
            {
                label: 'Gia hạn',
                to: '#/task/extend',
            },
            {
                label: 'Hồ sơ hủy',
                to: '#/task/cancel',
            },
            {
                label: 'Điều chỉnh phê duyệt',
                to: '#/task/change',
            },
            {
                label: 'Phối hợp',
                to: '#/task/combination',
            },
            {
                label: 'Tìm kiếm',
                to: '#/task/search',
            },
            {
                label: 'Sao gửi mặc định',
                to: '#/task/usercc-default',
            },
            {
                label: 'Kiểm soát hồ sơ',
                to: '#/task/assign',
            },
        ],
    },
    {
        icon: 'pe-7s-folder',
        label: 'Quản lý văn bản',
        content: [
            {
                label: 'Văn bản vi phạm',
                to: '#/dms/dms2',
            },
            {
                label: 'Tìm kiếm',
                to: '#/dms',
            }
        ],
    },
    {
        icon: 'pe-7s-date',
        label: 'Kế hoạch công việc',
        content: [
            {
                label: 'Kế hoạch công việc',
                to: '#/khcv',
            },
            {
                label: 'KH ban hành văn bản',
                to: '#/vbqc',
            },
            {
                label: 'Báo cáo định kỳ',
                to: '#/bcdk',
            },
            {
                label: 'Mã công việc',
                to: '#/jd',
            }
        ],
    },
    {
        icon: 'pe-7s-graph3',
        label: 'Báo cáo thống kê',
        content: [
            {
                label: 'Tổng hợp PGV',
                to: '#/genaralPGV',
            },
            {
                label: 'Báo cáo HO',
                to: '#/AppKpiChart',
            },
            {
                label: 'KPIs',
                to: '#/kpis',
            },
            {
                label: 'Nhắc việc',
                to: '#/remindWork',
            },
            {
                label: 'Báo cáo tiến độ công việc',
                to: '#/report-status-application',
            },
            {
                label: 'Báo cáo tổng hợp tiến độ',
                to: '#/report-aggregate',
            },
            {
                label: 'Trạng thái và JD hồ sơ ',
                to: '#/report-progress-application',
            },


        ],
    }
];